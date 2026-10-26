"""Компенсация: обратная кривая к измеренной даёт линейный результат."""
import os
import sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from hybridrip.compensation import invert, parse, DEFAULT_STEPS


def test_inverse_linearizes():
    t = np.array(DEFAULT_STEPS) / 100
    for e in (1.3, 1.6, 2.0):
        gain = lambda x: 1 - (1 - x) ** e
        c = np.array(invert(t, gain(t)))
        g = np.linspace(0.03, 1, 200)
        res = gain(np.interp(g, c[:, 0], c[:, 1]))
        assert np.abs(res - g).max() < 0.01, e


def test_noisy_input_monotone():
    t = np.array([0, .1, .2, .3, .5, .7, 1])
    m = np.array([0, .15, .29, .28, .64, .83, 1])   # шум: 30 % < 20 %
    c = np.array(invert(t, m))
    assert np.all(np.diff(c[:, 1]) >= 0)


def test_example_file():
    p = os.path.join(os.path.dirname(__file__), "..", "compensation", "EXAMPLE_synthetic.csv")
    d = parse(p)
    assert set(d["curves"]) == {"am", "fm", "line"}
    assert abs(d["min_dots"]["fm"] - 0.005) < 1e-9


if __name__ == "__main__":
    test_inverse_linearizes(); test_noisy_input_monotone(); test_example_file(); print("OK")
