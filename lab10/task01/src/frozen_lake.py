"""Frozen Lake Environment

ToyText category — FrozenLake-v1
Stan gry (observation): DYSKRETNY — liczba całkowita 0..15 (pozycja na siatce 4x4)
Zestaw akcji (action):  DYSKRETNY — {0: lewo, 1: dół, 2: prawo, 3: góra}
=> TYP 1: Stan dyskretny, Akcja dyskretna
Cel: przejść z pola startowego (0) do celu (15) bez wpadnięcia w dziurę.
"""

import gymnasium as gym


def run_frozen_lake(n_episodes: int = 5, seed: int = 42) -> None:
    # render_mode=None — headless, bezpieczne na Windows bez serwera graficznego
    env = gym.make("FrozenLake-v1", is_slippery=True, render_mode=None)
    print("=" * 55)
    print("FROZEN LAKE — ToyText")
    print(f"  Przestrzeń stanów (observation_space): {env.observation_space}")
    print(f"  Przestrzeń akcji  (action_space):      {env.action_space}")
    print("  TYP 1: Stan DYSKRETNY | Akcja DYSKRETNA")
    print("=" * 55)
    total_reward = 0.0
    successes = 0
    for episode in range(n_episodes):
        _obs, _info = env.reset(seed=seed + episode)
        done = False
        ep_reward = 0.0
        steps = 0
        while not done:
            # Losowa próbka akcji z przestrzeni akcji
            action = env.action_space.sample()
            _obs, reward, terminated, truncated, _info = env.step(action)
            reward = float(reward)
            ep_reward += reward
            steps += 1
            done = terminated or truncated
        total_reward += ep_reward
        if ep_reward > 0:
            successes += 1
        print(f"  Episode {episode + 1:2d} | steps={steps:3d} | reward={ep_reward:.1f}")
    env.close()
    print(f"\n  Łącznie nagród: {total_reward:.1f} | Sukcesy: {successes}/{n_episodes}")


if __name__ == "__main__":
    run_frozen_lake()
