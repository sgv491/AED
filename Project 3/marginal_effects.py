import numpy as np
from typing import Callable, Tuple

import w8_estimation as est


LinkFunction = Callable[[np.ndarray], np.ndarray]
DerivativeFunction = Callable[[np.ndarray], np.ndarray]


def _ensure_row(x: np.ndarray) -> np.ndarray:
    """Return x as a 1xK row vector."""
    arr = np.asarray(x, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    return arr


def _delta_se(func: Callable[[np.ndarray], np.ndarray],
              theta_hat: np.ndarray,
              cov: np.ndarray) -> float:
    """Generic delta-method standard error calculator."""
    grad = est.centered_grad(lambda th: np.atleast_1d(func(th)), theta_hat)
    cov_me = grad @ cov @ grad.T
    return float(np.sqrt(cov_me.squeeze()))


def discrete_effect(theta_hat: np.ndarray,
                    x_base: np.ndarray,
                    x_alt: np.ndarray,
                    link: LinkFunction) -> float:
    """Discrete change in probability when switching from x_base to x_alt."""
    x0 = _ensure_row(x_base)
    x1 = _ensure_row(x_alt)
    return float(link(x1 @ theta_hat) - link(x0 @ theta_hat))


def discrete_effect_delta(theta_hat: np.ndarray,
                          cov: np.ndarray,
                          x_base: np.ndarray,
                          x_alt: np.ndarray,
                          link: LinkFunction) -> Tuple[float, float]:
    """Discrete effect with delta-method standard error."""
    effect = discrete_effect(theta_hat, x_base, x_alt, link)

    def effect_func(theta: np.ndarray) -> np.ndarray:
        return np.array([discrete_effect(theta, x_base, x_alt, link)])

    se = _delta_se(effect_func, theta_hat, cov)
    return effect, se


def continuous_effect(theta_hat: np.ndarray,
                      x_point: np.ndarray,
                      var_idx: int,
                      link_prime: DerivativeFunction) -> float:
    """Derivative of Pr(y=1|x) w.r.t. a continuous regressor."""
    x_row = _ensure_row(x_point)
    z = float(x_row @ theta_hat)
    return float(link_prime(z) * theta_hat[var_idx])


def continuous_effect_delta(theta_hat: np.ndarray,
                            cov: np.ndarray,
                            x_point: np.ndarray,
                            var_idx: int,
                            link_prime: DerivativeFunction) -> Tuple[float, float]:
    """Continuous effect with delta-method standard error."""
    effect = continuous_effect(theta_hat, x_point, var_idx, link_prime)

    def effect_func(theta: np.ndarray) -> np.ndarray:
        return np.array([continuous_effect(theta, x_point, var_idx, link_prime)])

    se = _delta_se(effect_func, theta_hat, cov)
    return effect, se
