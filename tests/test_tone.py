"""Проверка главного свойства: в любой смеси растров покрытие = тону.
Запуск:  python -m pytest tests  (или python tests/test_tone.py)"""
import os
import sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from hybridrip.screens import periodic_threshold, FMScreen
from hybridrip.blend import mix_thresholds, EmpiricalMixer

N = 1024
Y, X = np.mgrid[0:N, 0:N].astype(np.float32)
AM = periodic_threshold(X, Y, dpi=2400, lpi=150, angle_deg=7.5, shape="euclid")
LN = periodic_threshold(X, Y, dpi=2400, lpi=150, angle_deg=7.5, shape="line")
FM = FMScreen(3).threshold(X, Y)
EM = EmpiricalMixer(AM, LN)
TONES = np.linspace(0.02, 0.98, 25)


def max_err(T):
    return max(abs(float((t > T).mean()) - t) for t in TONES)


def test_single_screens():
    for T in (AM, LN, FM):
        assert max_err(T) < 0.004


def test_pairs_and_triple():
    for a in np.linspace(0.05, 0.95, 7):
        f = lambda v: np.full((N, N), v, np.float32)
        assert max_err(mix_thresholds([AM, FM], np.stack([f(a), f(1 - a)]))) < 0.003
        assert max_err(mix_thresholds([AM, LN], np.stack([f(a), f(1 - a)]), {1: EM})) < 0.004
        w3 = np.stack([f(a * .6), f((1 - a) * .6), f(.4)])
        assert max_err(mix_thresholds([AM, LN, FM], w3, {1: EM})) < 0.004


if __name__ == "__main__":
    test_single_screens(); test_pairs_and_triple(); print("OK")
