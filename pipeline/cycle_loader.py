"""
주행 사이클 로더 — UDDS / HWFET 자동 로드
EPA 표준 사이클 데이터를 dt_sim 해상도로 생성
24시간(86400초) 지원
"""

import os
import numpy as np

_CYCLE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "dataset")


def _generate_udds_cycle(dt: float = 0.01) -> tuple:
    """
    UDDS (FTP-72) 근사 생성
    총 1369초, 평균 31.5 km/h, 최대 91.2 km/h, 거리 12.07 km
    dt=0.01s 해상도
    """
    t_total = 1369  # s
    n = int(t_total / dt)
    t = np.linspace(0, t_total, n)

    v_kmh = np.zeros(n)

    segments = [
        (0, 20, 0, 0),        # idle
        (20, 40, 0, 50),      # accel
        (40, 80, 50, 50),     # cruise
        (80, 100, 50, 0),     # decel
        (100, 130, 0, 0),     # idle
        (130, 160, 0, 60),    # accel
        (160, 250, 60, 60),   # cruise
        (250, 280, 60, 30),   # decel
        (280, 310, 30, 55),   # accel
        (310, 420, 55, 55),   # cruise
        (420, 450, 55, 0),    # decel
        (450, 480, 0, 0),     # idle
        (480, 510, 0, 48),    # accel
        (510, 590, 48, 48),   # cruise
        (590, 620, 48, 25),   # decel
        (620, 650, 25, 50),   # accel
        (650, 730, 50, 50),   # cruise
        (730, 760, 50, 0),    # decel
        (760, 800, 0, 0),     # idle
        (800, 830, 0, 72),    # accel
        (830, 930, 72, 72),   # cruise
        (930, 960, 72, 40),   # decel
        (960, 990, 40, 60),   # accel
        (990, 1050, 60, 60),  # cruise
        (1050, 1090, 60, 91), # accel
        (1090, 1140, 91, 91), # cruise
        (1140, 1200, 91, 30), # decel
        (1200, 1230, 30, 55), # accel
        (1230, 1310, 55, 55), # cruise
        (1310, 1350, 55, 0),  # decel
        (1350, 1369, 0, 0),   # idle
    ]

    for t_start, t_end, v_start, v_end in segments:
        mask = (t >= t_start) & (t < t_end)
        frac = np.zeros_like(t)
        duration = t_end - t_start
        if duration > 0:
            frac = np.clip((t - t_start) / duration, 0, 1)
        v_kmh[mask] = v_start + (v_end - v_start) * frac[mask]

    v_ms = v_kmh / 3.6
    a = np.gradient(v_ms, dt)
    a = np.clip(a, -4.0, 4.0)

    return v_ms, a, dt


def _generate_hwfet_cycle(dt: float = 0.01) -> tuple:
    """
    HWFET 근사 생성
    총 765초, 평균 77.7 km/h, 최대 96.4 km/h, 거리 16.51 km
    dt=0.01s 해상도
    """
    t_total = 765
    n = int(t_total / dt)
    t = np.linspace(0, t_total, n)

    v_kmh = np.zeros(n)

    segments = [
        (0, 15, 0, 0),          # idle
        (15, 55, 0, 85),        # accel to highway
        (55, 130, 85, 88),      # slight accel
        (130, 200, 88, 72),     # gentle decel
        (200, 250, 72, 90),     # accel
        (250, 370, 90, 90),     # cruise
        (370, 400, 90, 75),     # decel
        (400, 430, 75, 82),     # accel
        (430, 530, 82, 82),     # cruise
        (530, 560, 82, 68),     # gentle decel
        (560, 590, 68, 85),     # accel
        (590, 700, 85, 85),     # cruise
        (700, 750, 85, 0),      # decel to stop
        (750, 765, 0, 0),       # idle
    ]

    for t_start, t_end, v_start, v_end in segments:
        mask = (t >= t_start) & (t < t_end)
        duration = t_end - t_start
        if duration > 0:
            frac = np.clip((t - t_start) / duration, 0, 1)
            v_kmh[mask] = v_start + (v_end - v_start) * frac[mask]

    v_ms = v_kmh / 3.6
    a = np.gradient(v_ms, dt)
    a = np.clip(a, -4.0, 4.0)

    return v_ms, a, dt


def load_cycle(cycle_name: str, duration_s: int, dt: float = 0.01) -> tuple:
    """
    사이클을 duration_s 동안 반복 생성 (dt 해상도)
    반환: (v_ms, a_ms2) 전체 시간 배열
    최대 24시간(86400초) 지원
    """
    if cycle_name.upper() == "UDDS":
        v_one, a_one, _ = _generate_udds_cycle(dt)
    elif cycle_name.upper() == "HWFET":
        v_one, a_one, _ = _generate_hwfet_cycle(dt)
    else:
        raise ValueError(f"Unknown cycle: {cycle_name}")

    total_samples = int(duration_s / dt)
    cycle_len = len(v_one)

    repeats = (total_samples // cycle_len) + 2
    v_full = np.tile(v_one, repeats)[:total_samples]
    a_full = np.tile(a_one, repeats)[:total_samples]

    # 반복 경계에서 가속도 스무딩 (dt=0.01s 정밀)
    smooth_samples = int(1.0 / dt)  # 1초 분량 스무딩
    for i in range(1, repeats):
        boundary = i * cycle_len
        if boundary < total_samples:
            smooth_range = min(smooth_samples, total_samples - boundary)
            if smooth_range > 1 and boundary > 0:
                a_full[boundary:boundary + smooth_range] = np.linspace(
                    a_full[boundary - 1],
                    a_full[min(boundary + smooth_range, total_samples - 1)],
                    smooth_range
                )

    return v_full, a_full
