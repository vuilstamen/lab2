#!/usr/bin/env python3
"""
2D DEM: удар гранулированной пули по горизонтальной стенке.

Условия:
  - Второй закон Ньютона (F = ma)
  - Взаимодействие только с ближайшими соседями (физически, через проверку overlap)
  - Парное взаимодействие (сила i-j и j-i)
  - Интегрирование Velocity Verlet для свободных частиц
  - Прямое вычисление частица-частица (PP-метод)
  - Отколовшиеся куски стенки не могут её разрушать (только снаряд)

Снаряд:
  - абсолютно твёрдое тело (rigid body)
  - любая гранула снаряда, коснувшаяся стенки, становится отдельным осколком
"""

from __future__ import annotations

import atexit
import os
import shutil
import traceback
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from multiprocessing import Pool, cpu_count
from multiprocessing.shared_memory import SharedMemory
from typing import Any

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

# =========================
# Параметры
# =========================

SUBSTEPS_PER_FRAME = 5
NUMBER_OF_FRAMES = 200
GIF_FRAME_DURATION = 20  # мс

DT = 0.001
GRAVITY = np.array([0.0, -20.0], dtype=np.float64)

PARTICLE_MASS = 1.0    #масса гранулы
PARTICLE_RADIUS = 0.08  #radius granuli

# Базовый контакт (стенка–стенка и смешанные пары)
CONTACT_K = 12000.0
CONTACT_DAMPING = 1500.0  #koeff dempfir
FRICTION_MU = 0.55
TANGENTIAL_DAMPING = 4000.0

# Отрыв закреплённой гранулы стенки
WALL_BREAK_FORCE = 1000.0   #sila otriva stenki

# Отрыв гранулы снаряда (мгновенный при контакте со стенкой)
PROJECTILE_BREAK_FORCE = 1500.0 #sila otriva snaryada
PROJECTILE_CONTACT_MEMORY = 8

# Горизонтальная стенка (доли высоты области)
WALL_THICKNESS_FRAC = 0.02  #tolschina sloya
WALL_BOTTOM_FRAC = 0.4
WALL_ROWS = 3  #kolvo sloyev

# Пуля
BULLET_WIDTH = 0.3    #diam snaryada
BULLET_BODY_LEN = 0.70
BULLET_NOSE_LEN = 0.55
BULLET_INITIAL_SPEED = 40.0   # Скалярная скорость
PROJECTILE_PACK_SCALE = 0.92

KIND_WALL = 0
KIND_PROJECTILE = 1

FRAMES_DIR = "frames"
GIF_FILENAME = "shell14.gif"
EPS = 1e-12

# =========================
# Shared memory
# =========================

class SharedNDArray:
    def __init__(self, shape: tuple[int, ...], dtype=np.float64):
        self.shape = tuple(shape)
        self.dtype = np.dtype(dtype)
        self.nbytes = int(np.prod(self.shape) * self.dtype.itemsize)
        self.shm = SharedMemory(create=True, size=max(self.nbytes, 1))
        self.arr = np.ndarray(self.shape, dtype=self.dtype, buffer=self.shm.buf)

    def meta(self) -> tuple[str, tuple[int, ...], str]:
        return self.shm.name, self.shape, self.dtype.str

    def close(self) -> None:
        try: self.shm.close()
        except Exception: pass

    def unlink(self) -> None:
        try: self.shm.unlink()
        except Exception: pass

def attach_shared(name: str, shape: tuple[int, ...], dtype_str: str):
    shm = SharedMemory(name=name)
    arr = np.ndarray(shape, dtype=np.dtype(dtype_str), buffer=shm.buf)
    return shm, arr

# =========================
# Состояние воркера
# =========================

_W: dict[str, Any] = {}
_W_SHMS: list[SharedMemory] = []

def _worker_init(static: dict[str, Any]) -> None:
    global _W, _W_SHMS
    for s in _W_SHMS:
        try: s.close()
        except Exception: pass
    _W_SHMS = []
    _W = {"static": static}

    for key, (name, shape, dtype_str) in static["shared"].items():
        shm, arr = attach_shared(name, shape, dtype_str)
        _W[key] = arr
        _W_SHMS.append(shm)

def _worker_compute_bundle(task: dict[str, Any]) -> dict[str, Any]:
    try:
        n = int(_W["static"]["n_particles"])
        pos = _W["pos"]
        vel = _W["vel"]
        radius = _W["radius"]
        kind = _W["kind"]
        bound = _W["bound"]

        st = _W["static"]
        pair_i = st["pair_i"]
        pair_j = st["pair_j"]

        local_f = np.zeros((n, 2), dtype=np.float64)
        local_f_proj_on_wall = np.zeros((n, 2), dtype=np.float64)
        hit_wall = np.zeros(n, dtype=np.bool_)

        p0, p1 = task["pair_range"]
        if p1 > p0:
            pi = pair_i[p0:p1]
            pj = pair_j[p0:p1]

            r = pos[pj] - pos[pi]
            dist = np.linalg.norm(r, axis=1)
            min_dist = radius[pi] + radius[pj]

            tiny = dist <= EPS
            nvec = np.empty_like(r)
            dist_w = dist.copy()
            if np.any(~tiny):
                nvec[~tiny] = r[~tiny] / dist_w[~tiny, None]
            if np.any(tiny):
                nvec[tiny] = (1.0, 0.0)
                dist_w[tiny] = 0.0

            overlap = np.where(tiny, min_dist, min_dist - dist_w)
            active = overlap > 0.0

            # Не считаем контакт между связанными гранулами снаряда (Rigid Body)
            both_bound_proj = (kind[pi] == KIND_PROJECTILE) & (kind[pj] == KIND_PROJECTILE) & bound[pi] & bound[pj]
            active = active & ~both_bound_proj

            if np.any(active):
                i_a = pi[active]
                j_a = pj[active]
                n_a = nvec[active]
                ov = overlap[active]
                ki = kind[i_a]
                kj = kind[j_a]

                k = CONTACT_K
                c_n = CONTACT_DAMPING
                c_t = TANGENTIAL_DAMPING
                mu = FRICTION_MU

                rel_v = vel[j_a] - vel[i_a]
                rel_vn = np.einsum("ij,ij->i", rel_v, n_a)
                fn = np.maximum(k * ov - c_n * rel_vn, 0.0)
                force_n_on_i = -fn[:, None] * n_a

                v_rel_i = vel[i_a] - vel[j_a]
                v_rel_n = np.einsum("ij,ij->i", v_rel_i, n_a)
                v_rel_t = v_rel_i - v_rel_n[:, None] * n_a
                speed_t = np.linalg.norm(v_rel_t, axis=1)
                force_t_on_i = np.zeros_like(force_n_on_i)
                sliding = speed_t > EPS
                if np.any(sliding):
                    tdir = np.zeros_like(v_rel_t)
                    tdir[sliding] = v_rel_t[sliding] / speed_t[sliding, None]
                    ft_visc = c_t * speed_t
                    ft_max = mu * fn
                    ft = np.minimum(ft_visc, ft_max)
                    force_t_on_i[sliding] = -ft[sliding, None] * tdir[sliding]

                force_on_i = force_n_on_i + force_t_on_i
                np.add.at(local_f, i_a, force_on_i)
                np.add.at(local_f, j_a, -force_on_i)

                # Разрушение стенки только от снаряда
                proj_wall = (
                    ((ki == KIND_PROJECTILE) & (kj == KIND_WALL))
                    | ((ki == KIND_WALL) & (kj == KIND_PROJECTILE))
                )
                if np.any(proj_wall):
                    iw = i_a[proj_wall]
                    jw = j_a[proj_wall]
                    ki_pw = ki[proj_wall]
                    kj_pw = kj[proj_wall]

                    hit_wall[iw[ki_pw == KIND_PROJECTILE]] = True
                    hit_wall[jw[kj_pw == KIND_PROJECTILE]] = True

                    force_pw = force_on_i[proj_wall]
                    mask_i_wall = (ki_pw == KIND_WALL)
                    np.add.at(local_f_proj_on_wall, iw[mask_i_wall], force_pw[mask_i_wall])
                    mask_j_wall = (kj_pw == KIND_WALL)
                    np.add.at(local_f_proj_on_wall, jw[mask_j_wall], -force_pw[mask_j_wall])

        return {"ok": True, "force": local_f, "f_proj_to_wall": local_f_proj_on_wall, "hit_wall": hit_wall}
    except Exception as exc:
        return {"ok": False, "error": f"{exc.__class__.__name__}: {exc}", "trace": traceback.format_exc()}

# =========================
# Абсолютно твёрдое тело (Rigid Body)
# =========================

def step_rigid_body(world: "World", dt: float) -> None:
    idx = np.flatnonzero(world.projectile_bound & ~world.fixed)
    if idx.size == 0: return

    m_i = world.mass[idx]
    M = m_i.sum()
    if M <= EPS: return

    if not np.array_equal(idx, world.projectile_bound_prev_idx):
        world.projectile_cm_pos = (m_i[:, None] * world.pos[idx]).sum(axis=0) / M
        world.projectile_cm_vel = (m_i[:, None] * world.vel[idx]).sum(axis=0) / M
        world.projectile_bound_prev_idx = idx.copy()

    rel_pos = world.pos[idx] - world.projectile_cm_pos
    I = (m_i * (rel_pos[:, 0] ** 2 + rel_pos[:, 1] ** 2)).sum()
    if I <= EPS: I = 1.0

    F_net = world.force[idx].sum(axis=0)
    tau = (rel_pos[:, 0] * world.force[idx, 1] - rel_pos[:, 1] * world.force[idx, 0]).sum()

    world.projectile_cm_vel += (F_net / M) * dt
    world.projectile_omega += (tau / I) * dt
    world.projectile_cm_pos += world.projectile_cm_vel * dt
    world.projectile_theta += world.projectile_omega * dt

    cos_t = np.cos(world.projectile_theta)
    sin_t = np.sin(world.projectile_theta)
    rot_mat = np.array([[cos_t, -sin_t], [sin_t, cos_t]])

    r0 = world.projectile_rest_pos[idx]
    r_rot = (rot_mat @ r0.T).T

    world.pos[idx] = world.projectile_cm_pos + r_rot
    world.vel[idx] = world.projectile_cm_vel + world.projectile_omega * np.column_stack((-r_rot[:, 1], r_rot[:, 0]))

def detach_projectile_particles(world: "World") -> int:
    cand = (world.kind == KIND_PROJECTILE) & world.projectile_bound & (world.proj_hit_timer > 0) & (~world.fixed)
    if not np.any(cand): return 0

    mag = np.linalg.norm(world.force, axis=1)
    release = cand & (mag >= PROJECTILE_BREAK_FORCE)
    n_rel = int(np.count_nonzero(release))
    if n_rel == 0: return 0

    world.projectile_bound[release] = False
    return n_rel

# =========================
# World
# =========================

@dataclass
class World:
    shm_pos: SharedNDArray
    shm_vel: SharedNDArray
    shm_radius: SharedNDArray
    shm_fixed: SharedNDArray
    shm_kind: SharedNDArray
    shm_bound: SharedNDArray

    pos: np.ndarray
    vel: np.ndarray
    radius: np.ndarray
    fixed: np.ndarray
    kind: np.ndarray
    projectile_bound: np.ndarray

    acc: np.ndarray
    force: np.ndarray
    mass: np.ndarray

    pair_i: np.ndarray
    pair_j: np.ndarray
    proj_hit_timer: np.ndarray

    projectile_rest_pos: np.ndarray
    projectile_cm_pos: np.ndarray
    projectile_cm_vel: np.ndarray
    projectile_theta: float
    projectile_omega: float
    projectile_bound_prev_idx: np.ndarray

    width: float
    height: float
    detached_projectile_total: int = 0

    @property
    def n(self) -> int:
        return int(self.pos.shape[0])

    def close(self) -> None:
        for s in (self.shm_pos, self.shm_vel, self.shm_radius, self.shm_fixed, self.shm_kind, self.shm_bound):
            s.close(); s.unlink()

def _make_horizontal_wall(width: float, height: float, prad: float) -> tuple[np.ndarray, float, float, float]:
    y_bottom = height * WALL_BOTTOM_FRAC
    thickness = height * WALL_THICKNESS_FRAC
    y_top = y_bottom + thickness

    step = 2.0 * prad * 0.95
    x_left, x_right = prad, width - prad
    n_cols = max(2, int(np.floor((x_right - x_left) / step)) + 1)
    step_x = (x_right - x_left) / (n_cols - 1) if n_cols > 1 else step

    n_rows = max(2, WALL_ROWS)
    step_y = (y_top - y_bottom - 2.0 * prad) / (n_rows - 1) if n_rows > 1 else step
    step_y = max(step_y, step * 0.85)

    prad_eff = min(prad, 0.48 * min(step_x, step_y))

    positions: list[list[float]] = []
    y0 = y_bottom + prad_eff
    for row in range(n_rows):
        x_off = 0.5 * step_x if (row % 2 == 1) else 0.0
        y = y0 + row * step_y
        cols = n_cols if x_off == 0.0 else max(1, n_cols - 1)
        for col in range(cols):
            x = x_left + x_off + col * step_x
            if x < prad_eff - 1e-9 or x > width - prad_eff + 1e-9: continue
            if y < y_bottom or y > y_top: continue
            positions.append([x, y])

    if not positions:
        y_mid = 0.5 * (y_bottom + y_top)
        n_cols = max(2, int(width / (2.0 * prad_eff)))
        xs = np.linspace(prad_eff, width - prad_eff, n_cols)
        positions = [[float(x), float(y_mid)] for x in xs]

    return np.asarray(positions, dtype=np.float64), float(prad_eff), float(y_bottom), float(y_top)

def _bullet_local_radius(s: float, half_w: float, body_len: float, nose_len: float) -> float:
    total = body_len + nose_len
    if s < 0.0 or s > total: return -1.0
    if s <= body_len: return half_w
    t = (s - body_len) / max(nose_len, EPS)
    return half_w * (1.0 - t) ** 0.55

def _make_bullet(center_x: float, nose_y: float, prad: float, width_domain: float, height_domain: float) -> np.ndarray:
    half_w = 0.5 * BULLET_WIDTH
    body_len, nose_len = BULLET_BODY_LEN, BULLET_NOSE_LEN
    total_len = body_len + nose_len

    step = 2.0 * prad * PROJECTILE_PACK_SCALE
    dy = step * np.sqrt(3.0) * 0.5

    positions: list[list[float]] = []
    n_rows = int(np.ceil(total_len / dy)) + 2
    for row in range(n_rows + 1):
        s = row * dy
        if s > total_len + prad: break
        r_loc = _bullet_local_radius(s, half_w, body_len, nose_len)
        if r_loc < 0.35 * prad:
            if r_loc >= 0.0:
                y = nose_y + (total_len - s)
                x = center_x
                if prad <= x <= width_domain - prad and prad <= y <= height_domain - prad:
                    positions.append([x, y])
            continue

        y = nose_y + (total_len - s)
        x_off = 0.5 * step if (row % 2 == 1) else 0.0
        n_side = int(np.ceil(r_loc / step)) + 1
        for col in range(-n_side, n_side + 1):
            x = center_x + col * step + x_off
            if abs(x - center_x) <= r_loc + 0.05 * prad:
                if prad <= x <= width_domain - prad and prad <= y <= height_domain - prad:
                    positions.append([x, y])

    if not positions: positions.append([center_x, nose_y + 0.5 * total_len])

    pts = np.asarray(positions, dtype=np.float64)
    pts = np.unique(np.round(pts, 6), axis=0)
    return pts

def create_world(width: float, height: float, bullet_x: float, bullet_y: float, bullet_axis_angle_deg: float) -> World:
    wall_pos, prad, y_bottom, y_top = _make_horizontal_wall(width, height, PARTICLE_RADIUS)
    proj_pos = _make_bullet(bullet_x, bullet_y, prad, width, height)

    n_wall, n_proj = wall_pos.shape[0], proj_pos.shape[0]
    n = n_wall + n_proj

    positions = np.vstack([wall_pos, proj_pos])
    kinds = np.zeros(n, dtype=np.int8)
    kinds[n_wall:] = KIND_PROJECTILE

    fixed = np.zeros(n, dtype=bool)
    fixed[:n_wall] = True

    velocities = np.zeros((n, 2), dtype=np.float64)
    projectile_bound = np.zeros(n, dtype=bool)
    projectile_bound[n_wall:] = True

    shm_pos = SharedNDArray((n, 2), np.float64)
    shm_vel = SharedNDArray((n, 2), np.float64)
    shm_radius = SharedNDArray((n,), np.float64)
    shm_fixed = SharedNDArray((n,), np.bool_)
    shm_kind = SharedNDArray((n,), np.int8)
    shm_bound = SharedNDArray((n,), np.bool_)

    shm_radius.arr[:] = prad
    shm_fixed.arr[:] = fixed
    shm_kind.arr[:] = kinds
    shm_bound.arr[:] = projectile_bound

    mass = np.full(n, PARTICLE_MASS, dtype=np.float64)
    acc = np.zeros((n, 2), dtype=np.float64)
    force = np.zeros((n, 2), dtype=np.float64)

    ii, jj = np.triu_indices(n, k=1)
    pair_i = ii.astype(np.int32)
    pair_j = jj.astype(np.int32)
    proj_hit_timer = np.zeros(n, dtype=np.int32)

    projectile_rest_pos = np.zeros_like(positions)
    idx_proj = np.flatnonzero(kinds == KIND_PROJECTILE)
    
    cm_pos = np.zeros(2, dtype=np.float64)
    cm_vel = np.zeros(2, dtype=np.float64)
    theta, omega = 0.0, 0.0
    bound_prev_idx = np.array([], dtype=np.int32)

    if idx_proj.size > 0:
        m_proj = mass[idx_proj]
        M_proj = m_proj.sum()
        cm_pos = (m_proj[:, None] * positions[idx_proj]).sum(axis=0) / M_proj
        rest = positions[idx_proj] - cm_pos
        
        angle = np.radians(bullet_axis_angle_deg)
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        rot_mat = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
        rot_rest = (rot_mat @ rest.T).T
        
        nose_pos = np.array([bullet_x, bullet_y])
        dist_to_nose = np.linalg.norm(positions[idx_proj] - nose_pos, axis=1)
        nose_local_idx = np.argmin(dist_to_nose)
        
        new_cm = nose_pos - rot_rest[nose_local_idx]
        
        positions[idx_proj] = new_cm + rot_rest
        projectile_rest_pos[idx_proj] = rest
        
        v_proj = np.array([np.sin(angle) * BULLET_INITIAL_SPEED, -np.cos(angle) * BULLET_INITIAL_SPEED])
        velocities[idx_proj] = v_proj
        
        cm_pos, cm_vel = new_cm, v_proj
        theta, bound_prev_idx = angle, idx_proj.astype(np.int32).copy()

    shm_pos.arr[:] = positions
    shm_vel.arr[:] = velocities

    print(f"Частиц: стенка={n_wall}, пуля={n_proj}, всего={n}")
    print(f"Пуля: угол={bullet_axis_angle_deg}°, скорость={BULLET_INITIAL_SPEED}, абсолютно твёрдое тело")
    print(f"Интегратор: Velocity Verlet для свободных частиц")

    return World(
        shm_pos=shm_pos, shm_vel=shm_vel, shm_radius=shm_radius, shm_fixed=shm_fixed, shm_kind=shm_kind, shm_bound=shm_bound,
        pos=shm_pos.arr, vel=shm_vel.arr, radius=shm_radius.arr, fixed=shm_fixed.arr, kind=shm_kind.arr, projectile_bound=shm_bound.arr,
        acc=acc, force=force, mass=mass, pair_i=pair_i, pair_j=pair_j, proj_hit_timer=proj_hit_timer,
        projectile_rest_pos=projectile_rest_pos, projectile_cm_pos=cm_pos, projectile_cm_vel=cm_vel,
        projectile_theta=theta, projectile_omega=omega, projectile_bound_prev_idx=bound_prev_idx,
        width=float(width), height=float(height)
    )

# =========================
# Параллельный движок
# =========================

def chunk_ranges(total: int, n_chunks: int) -> list[tuple[int, int]]:
    if total <= 0: return [(0, 0)]
    n_chunks = max(1, min(n_chunks, total))
    base, extra = divmod(total, n_chunks)
    out, start = [], 0
    for i in range(n_chunks):
        size = base + (1 if i < extra else 0)
        end = start + size
        out.append((start, end))
        start = end
    return out

def _apply_boundary_contacts(w: World) -> None:
    width, height = w.width, w.height
    k, c_n, c_t, mu = CONTACT_K, CONTACT_DAMPING, TANGENTIAL_DAMPING, FRICTION_MU

    for i in range(w.n):
        if w.fixed[i]: continue
        pos, vel, force, radius = w.pos[i], w.vel[i], w.force[i], float(w.radius[i])
        trials = (
            (radius - pos[0], np.array([1.0, 0.0])),
            (pos[0] + radius - width, np.array([-1.0, 0.0])),
            (radius - pos[1], np.array([0.0, 1.0])),
            (pos[1] + radius - height, np.array([0.0, -1.0])),
        )
        for ov, nrm in trials:
            if ov <= 0.0: continue
            vn = float(np.dot(vel, nrm))
            fn = max(k * ov - c_n * vn, 0.0)
            force += fn * nrm
            v_t = vel - vn * nrm
            st = float(np.linalg.norm(v_t))
            if st > EPS:
                ft = min(c_t * st, mu * fn)
                force += -ft * (v_t / st)

def _release_broken_wall_particles(w: World, proj_on_wall_force: np.ndarray) -> int:
    if not np.any(w.fixed): return 0
    mag = np.linalg.norm(proj_on_wall_force, axis=1)
    candidates = w.fixed & (w.kind == KIND_WALL) & (mag >= WALL_BREAK_FORCE)
    n_release = int(np.count_nonzero(candidates))
    if n_release: w.fixed[candidates] = False
    return n_release

class ParallelEngine:
    def __init__(self, world: World, n_jobs: int):
        self.world = world
        self.n_jobs = max(1, int(n_jobs))
        self.pool: Pool | None = None
        self.released_wall_total = 0
        self.detached_proj_total = 0

        static = {
            "n_particles": world.n,
            "pair_i": world.pair_i.copy(),
            "pair_j": world.pair_j.copy(),
            "shared": {
                "pos": world.shm_pos.meta(), "vel": world.shm_vel.meta(),
                "radius": world.shm_radius.meta(), "fixed": world.shm_fixed.meta(),
                "kind": world.shm_kind.meta(), "bound": world.shm_bound.meta(),
            },
        }
        self._static = static
        self.pair_ranges = chunk_ranges(world.pair_i.shape[0], self.n_jobs)
        self.n_tasks = len(self.pair_ranges)

        if self.n_jobs > 1:
            self.pool = Pool(processes=self.n_jobs, initializer=_worker_init, initargs=(static,))

    def close(self) -> None:
        if self.pool is not None:
            self.pool.close(); self.pool.join(); self.pool = None

    def compute_forces(self) -> None:
        w = self.world
        w.force[:] = w.mass[:, None] * GRAVITY[None, :]

        if np.any(w.proj_hit_timer > 0):
            w.proj_hit_timer[w.proj_hit_timer > 0] -= 1

        tasks = [{"pair_range": self.pair_ranges[t]} for t in range(self.n_tasks)]
        if self.pool is None:
            _worker_init(self._static)
            results = [_worker_compute_bundle(t) for t in tasks]
        else:
            results = self.pool.map(_worker_compute_bundle, tasks, chunksize=1)

        hit_any = np.zeros(w.n, dtype=bool)
        proj_on_wall_force = np.zeros((w.n, 2), dtype=np.float64)
        
        for res in results:
            if not res.get("ok", False):
                raise RuntimeError("Ошибка воркера:\n" + str(res.get("error", "")) + "\n" + str(res.get("trace", "")))
            w.force += res["force"]
            proj_on_wall_force += res["f_proj_to_wall"]
            hit_any |= res["hit_wall"]

        if np.any(hit_any):
            w.proj_hit_timer[hit_any] = PROJECTILE_CONTACT_MEMORY

        _apply_boundary_contacts(w)
        self.released_wall_total += _release_broken_wall_particles(w, proj_on_wall_force)
        self.detached_proj_total += detach_projectile_particles(w)
        w.detached_projectile_total = self.detached_proj_total

        if np.any(w.fixed):
            w.force[w.fixed] = 0.0
            w.vel[w.fixed] = 0.0

        w.acc[:] = w.force / w.mass[:, None]
        if np.any(w.fixed): w.acc[w.fixed] = 0.0

# =========================
# Интегрирование Velocity Verlet
# =========================

def velocity_verlet_step(engine: ParallelEngine, dt: float) -> None:
    w = engine.world
    old_acc = w.acc.copy()

    # 1. Обновление позиций независимых свободных частиц (Verlet)
    free = ~w.fixed
    non_rigid = free & ~w.projectile_bound
    w.pos[non_rigid] += w.vel[non_rigid] * dt + 0.5 * w.acc[non_rigid] * (dt * dt)

    # 2. Обновление позиций твёрдого тела (снаряда) - симплектический эйлер для rigid body
    step_rigid_body(w, dt)

    # 3. Расчёт новых сил и ускорений при новых позициях
    engine.compute_forces()

    # 4. Обновление скоростей независимых частиц (Verlet)
    w.vel[non_rigid] += 0.5 * (old_acc[non_rigid] + w.acc[non_rigid]) * dt

    if np.any(w.fixed): w.vel[w.fixed] = 0.0

# =========================
# Кадры / GIF
# =========================

def prepare_frames_dir() -> None:
    if os.path.exists(FRAMES_DIR): shutil.rmtree(FRAMES_DIR)
    os.makedirs(FRAMES_DIR, exist_ok=True)

def _render_frame_worker(width: float, height: float, particle_pos: np.ndarray, particle_fixed: np.ndarray, particle_kind: np.ndarray, particle_bound: np.ndarray, frame_index: int) -> str:
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlim(0, width); ax.set_ylim(0, height); ax.set_aspect("equal")
    ax.set_title("Velocity Verlet: Пуля → стенка")
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.grid(True, alpha=0.3)

    wall, proj = particle_kind == KIND_WALL, particle_kind == KIND_PROJECTILE
    wall_fixed, wall_free = wall & particle_fixed, wall & ~particle_fixed
    proj_body, proj_chip = proj & particle_bound, proj & ~particle_bound

    if np.any(wall_fixed): ax.scatter(particle_pos[wall_fixed, 0], particle_pos[wall_fixed, 1], s=28, c="tab:gray", label="стена (держится)", zorder=3)
    if np.any(wall_free): ax.scatter(particle_pos[wall_free, 0], particle_pos[wall_free, 1], s=28, c="tab:blue", label="стена (оторвалась)", zorder=3)
    if np.any(proj_body): ax.scatter(particle_pos[proj_body, 0], particle_pos[proj_body, 1], s=30, c="tab:red", label="пуля (целое тело)", zorder=4)
    if np.any(proj_chip): ax.scatter(particle_pos[proj_chip, 0], particle_pos[proj_chip, 1], s=26, c="tab:orange", label="осколок", zorder=5)

    ax.legend(loc="upper right", fontsize=8)
    path = os.path.join(FRAMES_DIR, f"frame_{frame_index:04d}.png")
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)
    return path

def snapshot_for_render(world: World):
    return world.pos.copy(), world.fixed.copy(), world.kind.copy(), world.projectile_bound.copy()

def create_gif() -> None:
    files = sorted(os.path.join(FRAMES_DIR, f) for f in os.listdir(FRAMES_DIR) if f.lower().endswith(".png"))
    if not files: print("Нет PNG-кадров для GIF."); return
    frames = [Image.open(f).convert("RGB") for f in files]
    frames[0].save(GIF_FILENAME, save_all=True, append_images=frames[1:], duration=GIF_FRAME_DURATION, loop=0)
    for im in frames: im.close()

# =========================
# Симуляция
# =========================

def simulate(width: float, height: float, n_jobs: int, bullet_x: float, bullet_y: float, bullet_axis_angle_deg: float) -> None:
    world = create_world(width, height, bullet_x, bullet_y, bullet_axis_angle_deg)
    engine = ParallelEngine(world, n_jobs)
    prepare_frames_dir()

    def _cleanup() -> None:
        try: engine.close()
        finally: world.close()

    atexit.register(_cleanup)

    try:
        engine.compute_forces()
        render_workers = min(4, max(1, (os.cpu_count() or 2) // 2))
        futures = []
        with ThreadPoolExecutor(max_workers=render_workers) as renderer:
            for frame_idx in range(NUMBER_OF_FRAMES):
                for _ in range(SUBSTEPS_PER_FRAME):
                    velocity_verlet_step(engine, DT)

                ppos, pfix, pkind, pbound = snapshot_for_render(world)
                futures.append(renderer.submit(_render_frame_worker, world.width, world.height, ppos, pfix, pkind, pbound, frame_idx))

                wall, proj = world.kind == KIND_WALL, world.kind == KIND_PROJECTILE
                wall_free = int(np.count_nonzero(wall & ~world.fixed))
                proj_chip = int(np.count_nonzero(proj & ~world.projectile_bound))
                proj_body = proj & world.projectile_bound

                if np.any(proj_body):
                    m = world.mass[proj_body]
                    cm = (m[:, None] * world.pos[proj_body]).sum(axis=0) / m.sum()
                    v_cm = (m[:, None] * world.vel[proj_body]).sum(axis=0) / m.sum()
                elif np.any(proj):
                    cm = world.pos[proj].mean(axis=0); v_cm = world.vel[proj].mean(axis=0)
                else: cm = v_cm = np.array([0.0, 0.0])

                print(f"Кадр {frame_idx + 1}/{NUMBER_OF_FRAMES} | стена↓ {wall_free} (Σ {engine.released_wall_total}) | пуля-осколки {proj_chip} (Σ {engine.detached_proj_total}) | CM=({cm[0]:.3f},{cm[1]:.3f}) v=({v_cm[0]:.3f},{v_cm[1]:.3f})")

            for fut in futures: fut.result()

        create_gif()
        print(f"\nGIF: {GIF_FILENAME}\nКадры: {FRAMES_DIR}/")
    finally:
        try: atexit.unregister(_cleanup)
        except Exception: pass
        _cleanup()

# =========================
# Ввод
# =========================

def ask_positive_float(prompt: str) -> float:
    while True:
        try:
            v = float(input(prompt).replace(",", "."))
            if v > 0: return v
            print("Нужно положительное число.")
        except ValueError: print("Ошибка: введите число.")

def ask_float(prompt: str) -> float:
    while True:
        try: return float(input(prompt).replace(",", "."))
        except ValueError: print("Ошибка: введите число.")

def ask_workers() -> int:
    cpus = cpu_count() or 1
    print(f"Логических CPU: {cpus}")
    while True:
        try:
            v = int(input(f"Число процессов-воркеров [1..{cpus}]: "))
            if 1 <= v <= cpus: return v
            print(f"Введите целое от 1 до {cpus}.")
        except ValueError: print("Ошибка: целое число.")

def ask_bullet_position(width: float, height: float) -> tuple[float, float, float]:
    print("\nНачальное положение ПУЛИ:")
    print("  Угол: 0 = вниз, 90 = вправо, -90 = влево, 180 = вверх")
    while True:
        x, y, angle = ask_float("  X носа: "), ask_float("  Y носа: "), ask_float("  Угол оси (градусы): ")
        if 0.0 <= x <= width and 0.0 <= y <= height: return x, y, angle
        print("  Координаты вне диапазона области.")

def main() -> None:
    print("DEM: абсолютно твёрдая пуля под углом → горизонтальная стенка (Velocity Verlet)")
    width, height = ask_positive_float("  Ширина: "), ask_positive_float("  Высота: ")
    bullet_x, bullet_y, bullet_angle = ask_bullet_position(width, height)
    n_jobs = ask_workers()
    simulate(width, height, n_jobs, bullet_x, bullet_y, bullet_angle)

if __name__ == "__main__":
    main()