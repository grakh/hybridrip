"""
Углы растра по сепарациям.

Порядок выбора угла для сепарации:
  1. таблица задания (cfg["angles"]) — точное имя или шаблон со * («PANTONE 485*»);
  2. библиотека плашечных цветов (settings.json → "spot_angles"), те же правила;
  3. для триады C / M / Y / K (в т.ч. Cyan, Black…) — углы триады из конфига;
  4. для прочих плашек — по очереди из списка spot_default_angles.
Имена сравниваются без учёта регистра и лишних пробелов.
"""
from __future__ import annotations
import fnmatch
import re

DEFAULT_PROCESS = {"C": 67.5, "M": 37.5, "Y": 82.5, "K": 7.5}
DEFAULT_SPOT_CYCLE = [22.5, 52.5, 82.5, 7.5, 37.5, 67.5]
PRESETS = ["0", "7.5", "15", "22.5", "30", "37.5", "45", "52.5", "60", "67.5", "75", "82.5"]
_ALIAS = {"c": "C", "cyan": "C", "m": "M", "magenta": "M",
          "y": "Y", "yellow": "Y", "k": "K", "black": "K"}


def norm(name: str) -> str:
    return re.sub(r"\s+", " ", str(name)).strip().lower()


def process_key(name: str):
    """'Cyan' -> 'C' и т.п.; для плашек None."""
    return _ALIAS.get(norm(name))


def _lookup(name: str, table: dict):
    if not table:
        return None
    n = norm(name)
    for k, v in table.items():            # сначала точное совпадение
        if norm(k) == n:
            return float(v)
    for k, v in table.items():            # затем шаблоны со *
        if "*" in k or "?" in k:
            if fnmatch.fnmatchcase(n, norm(k)):
                return float(v)
    return None


def angle_for(name: str, idx: int, cfg: dict, spot_library: dict | None = None):
    """Возвращает (угол, источник)."""
    job = cfg.get("angles", {}) or {}
    a = _lookup(name, job)
    if a is not None:
        return a % 180, "задание"
    pk = process_key(name)
    if pk is None:
        a = _lookup(name, spot_library or cfg.get("spot_angles") or {})
        if a is not None:
            return a % 180, "библиотека"
    if pk is not None:
        v = job.get(pk, DEFAULT_PROCESS[pk])
        return float(v) % 180, "триада"
    cycle = cfg.get("spot_default_angles") or DEFAULT_SPOT_CYCLE
    return float(cycle[idx % len(cycle)]) % 180, "по умолчанию"


def conflicts(assign: dict, tol: float = 0.5):
    """Пары сепараций с одинаковым углом (с точностью tol, по модулю 90° для круглой точки)."""
    names = list(assign)
    out = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            d = abs((assign[names[i]] - assign[names[j]]) % 90.0)
            if min(d, 90.0 - d) < tol:
                out.append((names[i], names[j]))
    return out
