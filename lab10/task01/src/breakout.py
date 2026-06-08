"""Breakout Environment

-------------------
Atari category — ALE/Breakout-v5
Stan gry (observation): CIĄGŁY (piksele) — Box(210, 160, 3) obraz RGB
Zestaw akcji (action):  DYSKRETNY — {0: NOOP, 1: FIRE, 2: RIGHT, 3: LEFT}
=> TYP 2: Stan ciągły (przestrzeń pikselowa), Akcja dyskretna
Cel: rozbić jak najwięcej cegieł piłką (reward = liczba rozbitych cegieł).
Wymaga: gymnasium[atari] + gymnasium[accept-rom-license]
"""

import ale_py
import gymnasium as gym

# W gymnasium >= 1.0 Atari wymaga ręcznej rejestracji środowisk ale_py
gym.register_envs(ale_py)


def run_breakout(n_episodes: int = 3, seed: int = 42) -> None:
    env = gym.make("ALE/Breakout-v5", render_mode=None)
    print("=" * 60)
    print("BREAKOUT — Atari")
    print(f"  Przestrzeń stanów (observation_space): {env.observation_space}")
    print(f"  Przestrzeń akcji  (action_space):      {env.action_space}")
    print("  TYP 2: Stan CIĄGŁY (piksele RGB) | Akcja DYSKRETNA")
    print("=" * 60)
    for episode in range(n_episodes):
        _obs, _info = env.reset(seed=seed + episode)
        done = False
        ep_reward = 0.0
        steps = 0
        while not done and steps < 500:  # ograniczamy do 500 kroków
            # Losowa próbka akcji z przestrzeni akcji (4 przyciski Atari)
            action = env.action_space.sample()
            _obs, reward, terminated, truncated, _info = env.step(action)
            reward = float(reward)
            ep_reward += reward
            steps += 1
            done = terminated or truncated
        print(f"  Episode {episode + 1:2d} | steps={steps:3d} | score={ep_reward:.0f}")
    env.close()


if __name__ == "__main__":
    run_breakout()
