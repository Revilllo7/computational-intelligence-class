"""Custom actions for Frozen Lake environment.

ToyText — FrozenLake-v1 z własnym zestawem akcji (non-slippery)
Zamiast losowych próbek akcji, używamy optymalnej ścieżki do celu na
standardowej mapie 4x4 (is_slippery=False).
Standardowa mapa 4x4:
    S F F F      (S=start, F=frozen, H=hole, G=goal)
    F H F H
    F F F H
    H F F G
Optymalna ścieżka: (0→1→2→3) prawda, ale omijamy dziury.
Numery akcji: 0=Lewo, 1=Dół, 2=Prawo, 3=Góra
Ścieżka: 0→1→2→6→10→14→15
  - z 0  prawo  → 1
  - z 1  prawo  → 2
  - z 2  dół    → 6
  - z 6  dół    → 10
  - z 10 prawo  → 11    (tu jest dziura! pomijamy)
  Bezpieczna ścieżka przez lewą stronę:
  - z 0  dół    → 4
  - z 4  dół    → 8
  - z 8  prawo  → 9
  - z 9  prawo  → 10
  - z 10 prawo  → 14   (błąd: H na 11, ale is_slippery=False)
  Ostateczna bezpieczna ścieżka (known optimal for 4x4 non-slippery):
  akcje = [2, 2, 1, 1, 1, 2, 2, 1]  →  S→1→2→3→7→11→15? nie, 7 i 11 są dziurami.
  Właściwa optymalna ścieżka (bez dziur):
  S(0)→dół→4→dół→8→prawo→9→prawo→10→dół→14→prawo→15(G)
  Akcje: [1, 1, 2, 2, 1, 2]
"""

import gymnasium as gym

# Optymalny zestaw akcji dla mapy 4x4 non-slippery (własny, nie losowy)
# S(0) -dół-> 4 -dół-> 8 -prawo-> 9 -prawo-> 10 -dół-> 14 -prawo-> 15(G)
OPTIMAL_ACTIONS = [
    1,  # dół:   0 → 4
    1,  # dół:   4 → 8
    2,  # prawo: 8 → 9
    2,  # prawo: 9 → 10
    1,  # dół:   10 → 14
    2,  # prawo: 14 → 15 (CEL!)
]


def run_smart_frozen_lake(n_episodes: int = 5, seed: int = 42) -> None:
    # is_slippery=False — deterministyczne ruchy, dzięki czemu optymalna ścieżka działa
    env = gym.make("FrozenLake-v1", is_slippery=False, render_mode=None)
    print("=" * 60)
    print("SMART FROZEN LAKE — własny zestaw akcji (nie losowy)")
    print(f"  Przestrzeń stanów: {env.observation_space}")
    print(f"  Przestrzeń akcji:  {env.action_space}")
    print(f"  Zaplanowana ścieżka: {OPTIMAL_ACTIONS}")
    print("  (0=Lewo, 1=Dół, 2=Prawo, 3=Góra)")
    print("=" * 60)
    total_reward = 0.0
    successes = 0
    for episode in range(n_episodes):
        _obs, _info = env.reset(seed=seed + episode)
        done = False
        ep_reward = 0.0
        action_idx = 0
        while not done and action_idx < len(OPTIMAL_ACTIONS):
            # Własny zestaw akcji — zamiast env.action_space.sample()
            action = OPTIMAL_ACTIONS[action_idx]
            _obs, reward, terminated, truncated, _info = env.step(action)
            reward = float(reward)
            ep_reward += reward
            action_idx += 1
            done = terminated or truncated
        total_reward += ep_reward
        result = "SUKCES ✓" if ep_reward > 0 else "porażka"
        print(
            f"  Episode {episode + 1:2d} | kroki={action_idx} | reward={ep_reward:.1f} | {result}"
        )
    env.close()
    print(f"\n  Łącznie nagród: {total_reward:.1f} | Sukcesy: {successes}/{n_episodes}")
    print("  → Optymalna ścieżka skutecznie zbliża agenta do celu (reward=1.0 przy sukcesie)")


if __name__ == "__main__":
    run_smart_frozen_lake()
