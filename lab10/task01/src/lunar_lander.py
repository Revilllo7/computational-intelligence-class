"""Lunar Lander Environment

Box2D category — LunarLander-v3
Stan gry (observation): CIĄGŁY — Box(8,) [x, y, vx, vy, kąt, v_kąt, lewa noga, prawa noga]
Zestaw akcji (action):  DYSKRETNY — {0: nic, 1: silnik lewy, 2: silnik główny, 3: silnik prawy}
=> TYP 2: Stan ciągły, Akcja dyskretna
Cel: miękko wylądować między flagami (reward >= 200 = sukces).
Wymaga: swig + gymnasium[box2d]
"""

import gymnasium as gym


def run_lunar_lander(n_episodes: int = 5, seed: int = 42) -> None:
    env = gym.make("LunarLander-v3", render_mode=None)
    print("=" * 55)
    print("LUNAR LANDER — Box2D")
    print(f"  Przestrzeń stanów (observation_space): {env.observation_space}")
    print(f"  Przestrzeń akcji  (action_space):      {env.action_space}")
    print("  TYP 2: Stan CIĄGŁY | Akcja DYSKRETNA")
    print("=" * 55)
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
        result = "SUKCES" if ep_reward >= 200 else "porażka"
        print(f"  Episode {episode + 1:2d} | steps={steps:3d} | reward={ep_reward:7.2f} [{result}]")
    env.close()


if __name__ == "__main__":
    run_lunar_lander()
