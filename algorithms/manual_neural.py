from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class NetworkState:
    # Container for a tiny fully connected 2-2-1 network state.
    # Weight convention for w_input_hidden is (hidden, input), i.e.
    # # [[w_h1_x1, w_h1_x2], [w_h2_x1, w_h2_x2]].

    w_input_hidden: np.ndarray
    b_hidden: np.ndarray
    w_hidden_output: np.ndarray
    b_output: float


def sigmoid(x: np.ndarray | float) -> np.ndarray | float:
    # `Sigmoid activation function.
    return 1.0 / (1.0 + np.exp(-x))


def mse_loss(y_pred: float, y_true: float) -> float:
    # Mean-squared error for one output: 0.5 * (y_pred - y_true)^2.
    return 0.5 * float((y_pred - y_true) ** 2)


def forward_pass(x: np.ndarray, state: NetworkState) -> dict[str, np.ndarray | float]:
    # Forward pass through a fixed 2-2-1 network.
    # Hidden layers use sigmoid activation, output layer is linear.
    
    z_hidden = state.w_input_hidden @ x + state.b_hidden
    a_hidden = sigmoid(z_hidden)
    z_output = float(a_hidden @ state.w_hidden_output + state.b_output)
    y_pred = z_output

    return {
        "x": x,
        "z_hidden": z_hidden,
        "a_hidden": a_hidden,
        "z_output": z_output,
        "y_pred": y_pred,
    }


def backprop_step(
    cache: dict[str, np.ndarray | float],
    y_true: float,
    state: NetworkState,
    eta: float,
) -> tuple[NetworkState, dict[str, np.ndarray | float]]:
    # Run one gradient-descent step for 2-2-1 net with hidden sigmoid + linear output.
    x = cache["x"]
    a_hidden = cache["a_hidden"]
    y_pred = float(cache["y_pred"])

    # For linear output y=z_out, so dL/dz_out = dL/dy = (y - y_true).
    delta_out = y_pred - y_true

    grad_w_hidden_output = a_hidden * delta_out
    grad_b_output = delta_out

    delta_hidden = state.w_hidden_output * delta_out * a_hidden * (1.0 - a_hidden)
    grad_w_input_hidden = np.outer(delta_hidden, x)
    grad_b_hidden = delta_hidden

    new_state = NetworkState(
        w_input_hidden=state.w_input_hidden - eta * grad_w_input_hidden,
        b_hidden=state.b_hidden - eta * grad_b_hidden,
        w_hidden_output=state.w_hidden_output - eta * grad_w_hidden_output,
        b_output=float(state.b_output - eta * grad_b_output),
    )

    gradients = {
        "delta_out": delta_out,
        "grad_w_hidden_output": grad_w_hidden_output,
        "grad_b_output": grad_b_output,
        "delta_hidden": delta_hidden,
        "grad_w_input_hidden": grad_w_input_hidden,
        "grad_b_hidden": grad_b_hidden,
    }
    return new_state, gradients
