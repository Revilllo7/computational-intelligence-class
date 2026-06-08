"""Half Cheetah Environment

MuJoCo category — HalfCheetah-v5
Stan gry (observation): CIĄGŁY — Box(17,) [pozycje i prędkości stawów]
Zestaw akcji (action):  CIĄGŁY — Box(6,) momenty sił [-1, 1] dla 6 stawów
=> TYP 3: Stan ciągły, Akcja ciągła
Cel: biegać jak najszybciej do przodu.
Wymaga: gymnasium[mujoco]
"""

import gymnasium as gym


def run_half_cheetah(n_episodes: int = 3, seed: int = 42) -> None:
    env = gym.make("HalfCheetah-v5", render_mode=None)
    print("=" * 60)
    print("HALF CHEETAH — MuJoCo")
    print(f"  Przestrzeń stanów (observation_space): {env.observation_space}")
    print(f"  Przestrzeń akcji  (action_space):      {env.action_space}")
    print("  TYP 3: Stan CIĄGŁY | Akcja CIĄGŁA")
    print("=" * 60)
    for episode in range(n_episodes):
        _obs, _info = env.reset(seed=seed + episode)
        done = False
        ep_reward = 0.0
        steps = 0
        while not done and steps < 200:  # ograniczamy do 200 kroków na epizod
            # Losowa próbka ciągłej akcji (6 momentów sił)
            action = env.action_space.sample()
            _obs, reward, terminated, truncated, _info = env.step(action)
            reward = float(reward)
            ep_reward += reward
            steps += 1
            done = terminated or truncated
        print(f"  Episode {episode + 1:2d} | steps={steps:3d} | reward={ep_reward:8.2f}")
    env.close()


if __name__ == "__main__":
    run_half_cheetah()
