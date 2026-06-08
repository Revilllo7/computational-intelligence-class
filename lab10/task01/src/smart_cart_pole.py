"""Custom actions for Cart Pole environment.

Classic Control — CartPole-v1 z własnym zestawem akcji (heurystyka kąta)
Zamiast losowych próbek akcji, używamy prostej heurystyki:
  - Jeśli kąt słupka > 0 (słupek chyli się w prawo) → pchnij wózek w prawo (akcja 1)
  - Jeśli kąt słupka < 0 (słupek chyli się w lewo)  → pchnij wózek w lewo  (akcja 0)
Ta heurystyka balistyczna (ang. "bang-bang control") jest znacznie lepsza
od losowego doboru akcji i pozwala utrzymać słupek przez wiele kroków.
Obserwacja (obs): [pos_wózka, vel_wózka, kąt_słupka, vel_kąt]
  - obs[2] = kąt słupka (θ), dodatni = w prawo, ujemny = w lewo
"""

import gymnasium as gym


def heuristic_action(obs) -> int:
    """
    Własna heurystyka zamiast env.action_space.sample():
    Ważona kombinacja kąta słupka i jego prędkości kątowej.
    Uwzględniając prędkość kątową, heurystyka znacznie lepiej balansuje słupek
    (działa jak uproszczony kontroler PD — ang. Proportional-Derivative).
    obs[2] = kąt słupka (θ):          dodatni = w prawo, ujemny = w lewo
    obs[3] = prędkość kątowa (θ_dot): dodatnia = obraca się w prawo
    """
    pole_angle = obs[2]
    pole_angle_vel = obs[3]
    # Ważony sygnał: kąt ma wagę 1.0, prędkość kątowa 0.5
    signal = pole_angle + 0.5 * pole_angle_vel
    return 1 if signal > 0 else 0  # 1=prawo, 0=lewo


def run_smart_cart_pole(n_episodes: int = 5, seed: int = 42) -> None:
    env = gym.make("CartPole-v1", render_mode=None)
    print("=" * 60)
    print("SMART CART POLE — własny zestaw akcji (heurystyka kąta)")
    print(f"  Przestrzeń stanów: {env.observation_space}")
    print(f"  Przestrzeń akcji:  {env.action_space}")
    print("  Heurystyka: kąt > 0 → akcja 1 (prawo), kąt < 0 → akcja 0 (lewo)")
    print("=" * 60)
    for episode in range(n_episodes):
        obs, _info = env.reset(seed=seed + episode)
        done = False
        ep_reward = 0.0
        steps = 0
        while not done:
            # Własna akcja oparta na obserwacji — zamiast env.action_space.sample()
            action = heuristic_action(obs)
            obs, reward, terminated, truncated, _info = env.step(action)
            reward = float(reward)
            ep_reward += reward
            steps += 1
            done = terminated or truncated
        result = "MAX ★" if steps >= 500 else f"{steps} kroków"
        print(f"  Episode {episode + 1:2d} | reward={ep_reward:6.1f} | {result}")
    env.close()
    print("\n  → Heurystyka kątowa utrzymuje słupek znacznie dłużej niż losowe akcje!")
    print("    (losowo: ~10-20 kroków | heurystyka: typowo 500 kroków = sukces)")


if __name__ == "__main__":
    run_smart_cart_pole()
