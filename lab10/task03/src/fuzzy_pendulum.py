"""
task03/fuzzy_pendulum.py
=========================
Fuzzy (Mamdani) controller for Pendulum-v1 — gymnasium Classic Control.

Środowisko: Pendulum-v1
  Obserwacja (obs): [cos(θ), sin(θ), θ_dot]
  Akcja: τ ∈ [-2, 2]  — moment siły (torque) [N·m]

Cel: wyprostować wahadło w górę (θ=0) i utrzymać je.
"""

import math
import sys
from pathlib import Path

import gymnasium as gym
import numpy as np
import simpful as sf

from common.fuzzy_plots import plot_linguistic_variable

_ROOT = Path(__file__).parent.parent.parent
if _ROOT not in sys.path:
    sys.path.insert(0, str(_ROOT))

PI = math.pi

# ─────────────────────────────────────────────────────────────────────────────
#  1. ZMIENNE LINGWISTYCZNE
# ─────────────────────────────────────────────────────────────────────────────

FS = sf.FuzzySystem()

# Kąt θ ∈ [-π, π]
a_NL = sf.FuzzySet(points=[[-PI, 1], [-PI / 2, 1], [-PI / 4, 0]], term="NL")
a_NS = sf.FuzzySet(points=[[-PI / 2, 0], [-PI / 4, 1], [0, 0]], term="NS")
a_ZE = sf.FuzzySet(points=[[-PI / 4, 0], [0, 1], [PI / 4, 0]], term="ZE")
a_PS = sf.FuzzySet(points=[[0, 0], [PI / 4, 1], [PI / 2, 0]], term="PS")
a_PL = sf.FuzzySet(points=[[PI / 4, 0], [PI / 2, 1], [PI, 1]], term="PL")

LV_angle = sf.LinguisticVariable(
    [a_NL, a_NS, a_ZE, a_PS, a_PL],
    concept="Kąt wahadła θ [rad]",
    universe_of_discourse=[-PI, PI],
)
FS.add_linguistic_variable("angle", LV_angle)

# Prędkość kątowa θ_dot ∈ [-8, 8]
v_Neg = sf.FuzzySet(points=[[-8, 1], [-2, 1], [0, 0]], term="Neg")
v_Zer = sf.FuzzySet(points=[[-2, 0], [0, 1], [2, 0]], term="Zer")
v_Pos = sf.FuzzySet(points=[[0, 0], [2, 1], [8, 1]], term="Pos")

LV_vel = sf.LinguisticVariable(
    [v_Neg, v_Zer, v_Pos],
    concept="Prędkość kątowa θ_dot [rad/s]",
    universe_of_discourse=[-8, 8],
)
FS.add_linguistic_variable("angle_vel", LV_vel)

# Moment siły τ ∈ [-2, 2]
t_NL = sf.FuzzySet(points=[[-2, 1], [-1, 1], [0, 0]], term="NL")
t_NS = sf.FuzzySet(points=[[-1.5, 0], [-0.5, 1], [0, 0]], term="NS")
t_ZE = sf.FuzzySet(points=[[-0.5, 0], [0, 1], [0.5, 0]], term="ZE")
t_PS = sf.FuzzySet(points=[[0, 0], [0.5, 1], [1.5, 0]], term="PS")
t_PL = sf.FuzzySet(points=[[0, 0], [1, 1], [2, 1]], term="PL")

LV_torque = sf.LinguisticVariable(
    [t_NL, t_NS, t_ZE, t_PS, t_PL],
    concept="Moment siły τ [N·m]",
    universe_of_discourse=[-2, 2],
)
FS.add_linguistic_variable("torque", LV_torque)

# ─────────────────────────────────────────────────────────────────────────────
#  2. REGUŁY WNIOSKOWANIA
# ─────────────────────────────────────────────────────────────────────────────

rules = [
    # Wahadło mocno po lewej (ujemne) -> duży moment w prawo (dodatni)
    "IF (angle IS NL) AND (angle_vel IS Neg) THEN (torque IS PL)",
    "IF (angle IS NL) AND (angle_vel IS Zer) THEN (torque IS PL)",
    "IF (angle IS NL) AND (angle_vel IS Pos) THEN (torque IS PS)",
    # Wahadło lekko po lewej
    "IF (angle IS NS) AND (angle_vel IS Neg) THEN (torque IS PS)",
    "IF (angle IS NS) AND (angle_vel IS Zer) THEN (torque IS PS)",
    "IF (angle IS NS) AND (angle_vel IS Pos) THEN (torque IS ZE)",
    # Wahadło blisko pionu
    "IF (angle IS ZE) AND (angle_vel IS Neg) THEN (torque IS PL)",  # ucieka w lewo -> mocno w prawo
    "IF (angle IS ZE) AND (angle_vel IS Zer) THEN (torque IS ZE)",  # idealnie
    "IF (angle IS ZE) AND (angle_vel IS Pos) THEN (torque IS NL)",  # ucieka w prawo -> mocno w lewo
    # Wahadło lekko po prawej (dodatnie) -> mały moment w lewo (ujemny)
    "IF (angle IS PS) AND (angle_vel IS Neg) THEN (torque IS ZE)",
    "IF (angle IS PS) AND (angle_vel IS Zer) THEN (torque IS NS)",
    "IF (angle IS PS) AND (angle_vel IS Pos) THEN (torque IS NS)",
    # Wahadło mocno po prawej
    "IF (angle IS PL) AND (angle_vel IS Neg) THEN (torque IS NS)",
    "IF (angle IS PL) AND (angle_vel IS Zer) THEN (torque IS NL)",
    "IF (angle IS PL) AND (angle_vel IS Pos) THEN (torque IS NL)",
]

FS.add_rules(rules)


# ─────────────────────────────────────────────────────────────────────────────
#  3. WYKRESY ZMIENNYCH LINGWISTYCZNYCH
# ─────────────────────────────────────────────────────────────────────────────


def generate_plots(plots_dir: str) -> None:
    Path(plots_dir).mkdir(exist_ok=True, parents=True)
    plot_linguistic_variable(
        LV_angle,
        "Kąt wahadła θ [rad]",
        str(Path(plots_dir, "angle.png")),
        title="Zmienna lingwistyczna: kąt (angle)",
    )
    plot_linguistic_variable(
        LV_vel,
        "Prędkość kątowa θ_dot [rad/s]",
        str(Path(plots_dir, "angle_vel.png")),
        title="Zmienna lingwistyczna: prędkość kątowa (angle_vel)",
    )
    plot_linguistic_variable(
        LV_torque,
        "Moment siły τ [N·m]",
        str(Path(plots_dir, "torque.png")),
        title="Zmienna lingwistyczna: moment siły (torque)",
    )


# ─────────────────────────────────────────────────────────────────────────────
#  4. STEROWANIE
# ─────────────────────────────────────────────────────────────────────────────


def fuzzy_action(obs) -> float:
    cos_th, sin_th, th_dot = float(obs[0]), float(obs[1]), float(obs[2])
    theta = math.atan2(sin_th, cos_th)
    th_dot = float(np.clip(th_dot, -8.0, 8.0))

    FS.set_variable("angle", theta)
    FS.set_variable("angle_vel", th_dot)

    result = FS.Mamdani_inference(["torque"])
    tau = result["torque"]

    return float(np.clip(tau, -2.0, 2.0))


# ─────────────────────────────────────────────────────────────────────────────
#  5. PĘTLA SYMULACJI
# ─────────────────────────────────────────────────────────────────────────────


def run_fuzzy_pendulum(n_episodes: int = 5, max_steps: int = 200, seed: int = 42) -> None:
    env = gym.make("Pendulum-v1", render_mode=None)

    print("=" * 68)
    print("FUZZY CONTROLLER — Pendulum-v1  (Mamdani)")
    print(f"  Przestrzeń stanów: {env.observation_space}")
    print(f"  Przestrzeń akcji:  {env.action_space}")
    print(f"  Liczba reguł:      {len(rules)}")
    print("=" * 68)

    ep_rewards = []

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)
        total_reward = 0.0

        for _step in range(max_steps):
            action = fuzzy_action(obs)
            obs, reward, terminated, truncated, _ = env.step([action])
            reward = float(reward)
            total_reward += reward
            if terminated or truncated:
                break

        ep_rewards.append(total_reward)
        cos_f = float(obs[0])
        angle_final = math.degrees(math.atan2(float(obs[1]), cos_f))
        print(
            f"  Epizod {ep + 1:2d} | nagroda={total_reward:8.2f} | kąt końcowy={angle_final:+6.1f}°"
        )

    env.close()

    avg = float(np.mean(ep_rewards))
    print(f"\n  Średnia nagroda: {avg:.2f}")
    print("  Losowy agent: ~-1200  |  Baza fuzzy: ~-1000  |  Dobry: ~-200")


if __name__ == "__main__":
    plots_dir = Path(__file__).parent / "plots"
    print("Generowanie wykresów zmiennych lingwistycznych...")
    generate_plots(str(plots_dir))
    print()
    run_fuzzy_pendulum(n_episodes=5)
