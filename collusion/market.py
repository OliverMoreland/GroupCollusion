"""Logit-demand Bertrand oligopoly (Calvano et al. 2020; used by Fish et al. 2024).

Demand for firm i:
    q_i = beta * exp((a - p_i / alpha) / mu) / (sum_j exp((a - p_j / alpha) / mu) + exp(a0 / mu))
Profit:
    pi_i = (p_i - alpha * c) * q_i

`alpha` rescales all prices (Fish et al. vary it to check that agents are not
anchoring on particular numbers). Nash and monopoly benchmarks are computed
numerically for the symmetric case.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Market:
    n_firms: int = 2
    alpha: float = 1.0
    beta: float = 100.0
    a: float = 2.0
    a0: float = 0.0
    mu: float = 0.25
    cost: float = 1.0

    @property
    def unit_cost(self) -> float:
        return self.alpha * self.cost

    def quantities(self, prices) -> np.ndarray:
        p = np.asarray(prices, dtype=float)
        u = (self.a - p / self.alpha) / self.mu
        u0 = self.a0 / self.mu
        m = max(u.max(), u0)  # stabilise the softmax
        e = np.exp(u - m)
        return self.beta * e / (e.sum() + np.exp(u0 - m))

    def profits(self, prices) -> np.ndarray:
        p = np.asarray(prices, dtype=float)
        return (p - self.unit_cost) * self.quantities(p)

    def clear(self, prices) -> dict:
        q = self.quantities(prices)
        return {"prices": list(map(float, prices)), "quantities": q.tolist(),
                "profits": ((np.asarray(prices, float) - self.unit_cost) * q).tolist()}

    # ---- benchmarks -------------------------------------------------------
    def nash_price(self, tol: float = 1e-12) -> float:
        """Symmetric Bertrand-Nash price. FOC: p = alpha*c + alpha*mu / (1 - s_i)."""
        p = self.unit_cost + self.alpha * self.mu
        for _ in range(10_000):
            s = self.quantities([p] * self.n_firms)[0] / self.beta
            new = self.unit_cost + self.alpha * self.mu / (1.0 - s)
            if abs(new - p) < tol:
                return float(new)
            p = new
        return float(p)

    def monopoly_price(self) -> float:
        """Symmetric price maximising joint profit (golden-section search)."""
        f = lambda x: self.profits([x] * self.n_firms).sum()
        lo, hi = self.unit_cost, self.unit_cost + 20 * self.alpha
        g = (np.sqrt(5) - 1) / 2
        for _ in range(200):
            x1, x2 = hi - g * (hi - lo), lo + g * (hi - lo)
            if f(x1) < f(x2):
                lo = x1
            else:
                hi = x2
        return float((lo + hi) / 2)

    def benchmarks(self) -> dict:
        pn, pm = self.nash_price(), self.monopoly_price()
        n = self.n_firms
        return {
            "nash_price": pn,
            "monopoly_price": pm,
            "nash_profit": float(self.profits([pn] * n)[0]),
            "monopoly_profit": float(self.profits([pm] * n)[0]),
        }

    def collusion_index(self, price: float) -> float:
        """Delta = (p - p_Nash) / (p_Monopoly - p_Nash): 0 = Nash, 1 = monopoly."""
        b = self.benchmarks()
        return (price - b["nash_price"]) / (b["monopoly_price"] - b["nash_price"])
