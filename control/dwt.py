"""
DWT 부하 분해기 — Daubechies-4, J=3, 100Hz
수식 (3.39): P_req = P_low + P_high
P_low  → SAC 에이전트 입력
P_high → SC 자동 할당 (에이전트 제어 불가)
"""

import numpy as np
from collections import deque

try:
    import pywt
    HAS_PYWT = True
except ImportError:
    HAS_PYWT = False


class DWTDecomposer:
    """
    슬라이딩 윈도우 256 샘플 (2.56초 @ 100Hz)
    """

    def __init__(self, wavelet: str = 'db4', level: int = 3, window: int = 256):
        self.wavelet = wavelet
        self.level = level
        self.window = window
        self._buffer = deque(maxlen=window)
        self._initialized = False

    def reset(self):
        self._buffer.clear()
        self._initialized = False

    def decompose(self, p_req: float) -> tuple:
        """
        반환: (P_low [W], P_high [W])
        """
        self._buffer.append(p_req)

        if len(self._buffer) < self.window:
            # 버퍼 미충족 시: 전부 저주파로 처리
            return (p_req, 0.0)

        if not HAS_PYWT:
            # PyWavelets 없으면 단순 이동평균으로 대체
            arr = np.array(self._buffer)
            p_low = np.mean(arr[-32:])
            p_high = p_req - p_low
            return (p_low, p_high)

        arr = np.array(self._buffer)
        coeffs = pywt.wavedec(arr, self.wavelet, level=self.level)

        # 저주파 = 근사 계수만으로 재구성
        coeffs_low = [coeffs[0]] + [np.zeros_like(c) for c in coeffs[1:]]
        p_low_signal = pywt.waverec(coeffs_low, self.wavelet)

        # 마지막 샘플
        p_low = float(p_low_signal[-1]) if len(p_low_signal) > 0 else p_req
        p_high = p_req - p_low

        return (p_low, p_high)
