"""Cart Pole Environment

--------------------
Classic Control category — CartPole-v1
Stan gry (observation): CIĄGŁY — Box(4,) [pozycja, prędkość wózka, kąt, prędkość kątowa słupka]
Zestaw akcji (action):  DYSKRETNY — {0: w lewo, 1: w prawo}
=> TYP 2: Stan ciągły, Akcja dyskretna
Cel: utrzymać słupek w pozycji pionowej jak najdłużej (max 500 kroków).
"""

import gymnasium as gym


def run_cart_pole(n_episodes: int = 5, seed: int = 42) -> None:
    env = gym.make("CartPole-v1", render_mode=None)
    print("=" * 55)
    print("CART POLE — Classic Control")
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
        print(f"  Episode {episode + 1:2d} | steps={steps:3d} | reward={ep_reward:.1f}")
    env.close()


if __name__ == "__main__":
    run_cart_pole()
