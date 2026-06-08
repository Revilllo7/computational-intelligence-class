"""Pendulum Environment

Classic Control category — Pendulum-v1
Stan gry (observation): CIĄGŁY — Box(3,) [cos(kąt), sin(kąt), prędkość kątowa]
Zestaw akcji (action):  CIĄGŁY — Box(1,) moment siły [-2.0, 2.0]
=> TYP 3: Stan ciągły, Akcja ciągła
Cel: wyprostować wahadło w górę i utrzymać je.
"""

import gymnasium as gym


def run_pendulum(n_episodes: int = 5, seed: int = 42) -> None:
    env = gym.make("Pendulum-v1", render_mode=None)
    print("=" * 55)
    print("PENDULUM — Classic Control")
    print(f"  Przestrzeń stanów (observation_space): {env.observation_space}")
    print(f"  Przestrzeń akcji  (action_space):      {env.action_space}")
    print("  TYP 3: Stan CIĄGŁY | Akcja CIĄGŁA")
    print("=" * 55)
    for episode in range(n_episodes):
        _obs, _info = env.reset(seed=seed + episode)
        done = False
        ep_reward = 0.0
        steps = 0
        while not done:
            # Losowa próbka akcji ciągłej (moment siły)
            action = env.action_space.sample()
            _obs, reward, terminated, truncated, _info = env.step(action)
            reward = float(reward)
            ep_reward += reward
            steps += 1
            done = terminated or truncated
        print(f"  Episode {episode + 1:2d} | steps={steps:3d} | reward={ep_reward:.2f}")
    env.close()


if __name__ == "__main__":
    run_pendulum()
