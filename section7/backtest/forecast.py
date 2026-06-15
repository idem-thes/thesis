"""Per-minute forecast engine for the Section 7.2 PnL backtest.
"""

from __future__ import annotations

import numpy as np

from code_section7.state import ParamSet, State, bar_quantities
from code_section7.engine.methods import (
    MINUTES_PER_YEAR,
    _m4_propagate_step,
    _m4_noise_vector,
)
from code_section7.engine.ukf import (
    KalmanStats,
    merwe_sigma_points,
    predict_step,
    unscented_transform,
    update_step,
)

_Q_REG = np.diag([1e-10, 1e-12, 1e-12, 1e-12, 1e-12])


def theta_hat_to_param_set(theta_hat: np.ndarray) -> ParamSet:
    """Convert the engine's 9-vector (PARAM_ORDER) to a ``ParamSet``.
    """
    return ParamSet(
        beta0=float(theta_hat[4]),
        beta1=float(theta_hat[5]),
        beta2=float(theta_hat[6]),
        lam10=float(theta_hat[0]),
        lam11=float(theta_hat[1]),
        theta1=float(theta_hat[7]),
        lam20=float(theta_hat[2]),
        lam21=float(theta_hat[3]),
        theta2=float(theta_hat[8]),
    )


def expected_dsigma_drift(state: State, theta: ParamSet, sigma_sq_realized: float) -> float:
    """Eq-54 sigma-drift in 1/year units evaluated at the given state.
    """
    bq = bar_quantities(state, theta)
    drift = -theta.beta1 * bq.lam1 * bq.R1 + (theta.beta2 * bq.lam2 / 2.0) * (
        sigma_sq_realized - bq.R2
    ) / np.sqrt(max(bq.R2_nobar, 1e-8))
    return float(drift)


def forecast_m2(
    state: State,
    theta: ParamSet,
    sigma_sq_realized: float,
    dt_minutes: float,
) -> float:
    """Return E[dVX over dt_minutes] in VX points for the M2 method.

    """
    drift = expected_dsigma_drift(state, theta, sigma_sq_realized)
    return 100.0 * drift * dt_minutes / MINUTES_PER_YEAR


class M4Filter:
    """UKF wrapper for M4: 5D state ``(sigma, R_{1,0}, R_{1,1}, R_{2,0}, R_{2,1})``.
    """

    def __init__(
        self,
        theta: ParamSet,
        x0: np.ndarray,
        P0: np.ndarray,
        R_meas: np.ndarray,
        alpha: float = 1e-2,
        beta_param: float = 2.0,
        kappa: float = -2.0,
    ) -> None:
        self.theta = theta
        self.x: np.ndarray = x0.copy()
        self.P: np.ndarray = P0.copy()
        self.R_meas: np.ndarray = R_meas
        self.alpha = alpha
        self.beta_param = beta_param
        self.kappa = kappa
        self._stats = KalmanStats()

    def _theta_array(self) -> np.ndarray:
        """Reorder ParamSet fields to PARAM_ORDER for _m4_propagate_step/_m4_noise_vector.

        """
        t = self.theta
        return np.array(
            [
                t.lam10,
                t.lam11,
                t.lam20,
                t.lam21,
                t.beta0,
                t.beta1,
                t.beta2,
                t.theta1,
                t.theta2,
            ]
        )

    def _make_fx(self, dt_years: float):
        """Return a closure that propagates a 5-vector over dt_years."""
        theta_array = self._theta_array()
        dt = dt_years

        def fx(state: np.ndarray) -> np.ndarray:
            return _m4_propagate_step(state, dt, theta_array)

        return fx

    def step(self, realized_return: float, dt_minutes: float, sigma_obs: float) -> None:
        """One UKF predict + update step. Mutates ``self.x`` and ``self.P``.
        """
        dt_years = dt_minutes / MINUTES_PER_YEAR

        fx = self._make_fx(dt_years)
        theta_array = self._theta_array()

        # State-dependent process noise Q = g g^T evaluated at prior mean.
        g_vec = _m4_noise_vector(self.x, dt_years, theta_array)
        Q_state_dep = np.outer(g_vec, g_vec)

        # Predict.
        x_pred, P_pred, self._stats = predict_step(
            self.x,
            self.P,
            fx,
            Q_state_dep,
            _Q_REG,
            alpha=self.alpha,
            beta=self.beta_param,
            kappa=self.kappa,
            stats=self._stats,
        )

        # Measurement function: observe only sigma component.
        def hx(state: np.ndarray) -> np.ndarray:
            return state[0:1]

        z = np.array([float(sigma_obs)])

        # Update.
        x_new, P_new, _info, self._stats = update_step(
            x_pred,
            P_pred,
            hx,
            z,
            self.R_meas,
            alpha=self.alpha,
            beta=self.beta_param,
            kappa=self.kappa,
            stats=self._stats,
        )

        self.x = x_new
        self.P = P_new

    def forecast(self, dt_minutes: float) -> float:
        """Predict-only step over dt_minutes; does NOT mutate self.x or self.P.
        """
        dt_years = dt_minutes / MINUTES_PER_YEAR
        fx = self._make_fx(dt_years)
        theta_array = self._theta_array()

        # Use a fresh stats accumulator so we don't corrupt self._stats.
        tmp_stats = KalmanStats()

        # Generate sigma points.
        sigmas, Wm, Wc, _ = merwe_sigma_points(
            self.x,
            self.P,
            alpha=self.alpha,
            beta=self.beta_param,
            kappa=self.kappa,
            stats=tmp_stats,
        )

        # Propagate sigma points through fx.
        propagated = np.array([fx(s) for s in sigmas])

        # State-dependent Q at the current prior mean.
        g_vec = _m4_noise_vector(self.x, dt_years, theta_array)
        Q_state_dep = np.outer(g_vec, g_vec)

        # Unscented transform to get x_pred.
        x_pred, _P_pred = unscented_transform(propagated, Wm, Wc, noise_cov=Q_state_dep + _Q_REG)

        return float(x_pred[0])


def forecast_m4(filter: M4Filter, dt_minutes: float) -> float:
    """Return E[dVX over dt_minutes] in VX points for the M4 method.
    """
    return 100.0 * (filter.forecast(dt_minutes) - float(filter.x[0]))
