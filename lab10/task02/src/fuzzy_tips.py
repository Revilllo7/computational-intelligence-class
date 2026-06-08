"""
task02/fuzzy_tips.py
---------------------
Simpful (v2.12.0) — Mamdani fuzzy controller for restaurant tip calculation.
Zmienne wejściowe / Input variables:
  - quality  [0, 10]: jakość jedzenia (poor / average / good)
  - service  [0, 10]: jakość obsługi  (bad / decent / great)
Zmienna wyjściowa / Output variable:
  - tip      [0, 25]: napiwek w procentach (low / medium / high)
Reguły / Rules:
  1. IF (quality IS poor)   OR (service IS bad)   THEN (tip IS low)
  2. IF (service IS decent)                        THEN (tip IS medium)
  3. IF (quality IS good)   OR (service IS great)  THEN (tip IS high)
Defuzyfikacja: Centroid (Mamdani)
Wykresy zmiennych lingwistycznych zapisywane do task02/plots/
via modułu common/fuzzy_plots.py
"""

import sys
from pathlib import Path

import simpful as sf

from common.fuzzy_plots import plot_linguistic_variable

_ROOT = str(Path(__file__).parent.parent.parent.resolve())
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# ---------------------------------------------------------------------------
# 1. Inicjalizacja systemu rozmytego
# ---------------------------------------------------------------------------
FS = sf.FuzzySystem()
# ---------------------------------------------------------------------------
# 2. Definicja zmiennych lingwistycznych i zbiorów rozmytych
# ---------------------------------------------------------------------------
# --- Jakość jedzenia (quality) [0, 10] ---
q_poor = sf.FuzzySet(points=[[0, 1], [3, 1], [5, 0]], term="poor")
q_average = sf.FuzzySet(points=[[2, 0], [5, 1], [8, 0]], term="average")
q_good = sf.FuzzySet(points=[[5, 0], [7, 1], [10, 1]], term="good")
LV_quality = sf.LinguisticVariable(
    [q_poor, q_average, q_good], concept="Food quality", universe_of_discourse=[0, 10]
)
FS.add_linguistic_variable("quality", LV_quality)
# --- Jakość obsługi (service) [0, 10] ---
s_bad = sf.FuzzySet(points=[[0, 1], [3, 1], [5, 0]], term="bad")
s_decent = sf.FuzzySet(points=[[2, 0], [5, 1], [8, 0]], term="decent")
s_great = sf.FuzzySet(points=[[5, 0], [7, 1], [10, 1]], term="great")
LV_service = sf.LinguisticVariable(
    [s_bad, s_decent, s_great], concept="Service quality", universe_of_discourse=[0, 10]
)
FS.add_linguistic_variable("service", LV_service)
# --- Napiwek (tip) [0, 25] ---
t_low = sf.FuzzySet(points=[[0, 1], [5, 1], [12.5, 0]], term="low")
t_medium = sf.FuzzySet(points=[[5, 0], [12.5, 1], [20, 0]], term="medium")
t_high = sf.FuzzySet(points=[[12.5, 0], [20, 1], [25, 1]], term="high")
LV_tip = sf.LinguisticVariable(
    [t_low, t_medium, t_high], concept="Tip percentage", universe_of_discourse=[0, 25]
)
FS.add_linguistic_variable("tip", LV_tip)
# ---------------------------------------------------------------------------
# 3. Definicja reguł rozmytych
# ---------------------------------------------------------------------------
RULE1 = "IF (quality IS poor)   OR (service IS bad)   THEN (tip IS low)"
RULE2 = "IF (service IS decent)                        THEN (tip IS medium)"
RULE3 = "IF (quality IS good)   OR (service IS great)  THEN (tip IS high)"
FS.add_rules([RULE1, RULE2, RULE3])
# ---------------------------------------------------------------------------
# 4. Wywołanie systemu dla kilku przykładowych wejść
# ---------------------------------------------------------------------------
test_cases = [
    {"quality": 2.0, "service": 1.5, "desc": "Słabe jedzenie, zła obsługa"},
    {"quality": 5.0, "service": 5.0, "desc": "Średnie jedzenie i obsługa"},
    {"quality": 8.0, "service": 9.0, "desc": "Dobre jedzenie, świetna obsługa"},
    {"quality": 3.0, "service": 7.0, "desc": "Słabe jedzenie, dobra obsługa"},
]


def compute_tip(quality_val: float, service_val: float) -> float:
    """Oblicza napiwek metodą Mamdaniego (centroid)."""
    FS.set_variable("quality", quality_val)
    FS.set_variable("service", service_val)
    result = FS.Mamdani_inference(["tip"])
    return result["tip"]


# ---------------------------------------------------------------------------
# 5. Generowanie wykresów zmiennych lingwistycznych
# ---------------------------------------------------------------------------
def generate_plots() -> None:
    plots_dir = Path.joinpath(Path(__file__).parent, "plots")
    Path.mkdir(plots_dir, exist_ok=True)
    plot_linguistic_variable(
        lv=LV_quality,
        var_name="Food quality [0-10]",
        save_path=str(Path(plots_dir, "quality.png")),
        title="Linguistic variable: quality (jakość jedzenia)",
    )
    plot_linguistic_variable(
        lv=LV_service,
        var_name="Service quality [0-10]",
        save_path=str(Path(plots_dir, "service.png")),
        title="Linguistic variable: service (jakość obsługi)",
    )
    plot_linguistic_variable(
        lv=LV_tip,
        var_name="Tip [0-25 %]",
        save_path=str(Path(plots_dir, "tip.png")),
        title="Linguistic variable: tip (napiwek)",
    )


if __name__ == "__main__":
    print("=" * 60)
    print("FUZZY TIP CONTROLLER — Simpful Mamdani")
    print("Reguły:")
    print(f"  R1: {RULE1}")
    print(f"  R2: {RULE2}")
    print(f"  R3: {RULE3}")
    print("=" * 60)
    for tc in test_cases:
        tip_val = compute_tip(tc["quality"], tc["service"])
        print(
            f"  quality={tc['quality']:4.1f}, service={tc['service']:4.1f}"
            f"  →  tip={tip_val:5.2f}%   [{tc['desc']}]"
        )
    print("\nGenerowanie wykresów zmiennych lingwistycznych...")
    generate_plots()
    print("Gotowe! Wykresy zapisane w task02/plots/")
