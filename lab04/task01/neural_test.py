from __future__ import annotations

import sys
import numpy as np
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
	sys.path.insert(0, str(PROJECT_ROOT))

from algorithms import NetworkState, backprop_step, forward_pass, mse_loss


BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "output"

ETA = 0.1
TARGET = 0.8
EXPECTED_FORWARD = 0.234
FORWARD_TOLERANCE = 0.02


# manually given state
def user_given_state() -> NetworkState:
	return NetworkState(
		# Mapping: [[w_h1_x1, w_h1_x2], [w_h2_x1, w_h2_x2]]
		w_input_hidden=np.array([[0.2000, -0.3000], [-0.5000, 0.1000]], dtype=float),
		# Mapping: [b_h1, b_h2]
		b_hidden=np.array([0.4000, -0.2000], dtype=float),
		# Mapping: [w_h1_out, w_h2_out]
		w_hidden_output=np.array([0.3000, -0.4000], dtype=float),
		# Mapping: b_out
		b_output=0.2000,
	)



# Helper for printing
def format_weights(state: NetworkState) -> str:
	return (
		f"w_input_hidden=\n{state.w_input_hidden}\n"
		f"b_hidden={state.b_hidden}\n"
		f"w_hidden_output={state.w_hidden_output}\n"
		f"b_output={state.b_output:.6f}"
	)


def main() -> None:
	OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

	# Input vector [x1, x2].
	x = np.array([0.6, 0.1], dtype=float)
	state = user_given_state()

	print("=" * 70)
	print("TASK01 - MANUAL NEURAL NETWORK (2-2-1)")
	print("=" * 70)
	print("Input vector:", x)
	print("Initial weights and biases:")
	print(format_weights(state))
	print(f"\n")

	cache = forward_pass(x, state)
	y_pred = float(cache["y_pred"])
	loss = mse_loss(y_pred, TARGET)

	print(f"Forward output y_pred for x=[0.6, 0.1]: {y_pred:.6f}")
	print(f"MSE loss L=(1/2)*(y_pred-target)^2 with target={TARGET:.3f}: {loss:.6f}")
	print(
		f"Forward check against expected {EXPECTED_FORWARD:.3f}: "
		f"{'PASS' if abs(y_pred - EXPECTED_FORWARD) <= FORWARD_TOLERANCE else 'CHECK WEIGHTS'}"
	)
	print(f"\n")

	new_state, gradients = backprop_step(cache, TARGET, state, ETA)

	print(f"Learning rate eta: {ETA}")
	print("Gradient summary:")
	print(f"delta_out={float(gradients['delta_out']):.6f}")
	print(f"delta_hidden={gradients['delta_hidden']}")
	print(f"\n")

	print("Updated weights and biases after one backprop step:")
	print(format_weights(new_state))

	report_path = OUTPUT_DIR / "weight_update.txt"
	report_path.write_text(
		"Initial state\n"
		+ format_weights(state)
		+ "\n\n"
		+ f"forward_output={y_pred:.6f}\n"
		+ f"target={TARGET:.6f}\n"
		+ f"loss={loss:.6f}\n"
		+ f"eta={ETA:.6f}\n\n"
		+ "Updated state\n"
		+ format_weights(new_state)
		+ "\n",
		encoding="utf-8",
	)
	print(f"\n")
	print(f"Saved report to: {report_path}")


if __name__ == "__main__":
	main()
