# === IMPORTS: BUILT-IN ===
from dataclasses import dataclass
from typing import Dict, Any, List

# === IMPORTS: THIRD-PARTY ===
import numpy as np
from numpy.linalg import inv
from scipy.stats import multivariate_normal


@dataclass
class ObservedDataset:
    Theta_obs: np.ndarray
    Thetas: List[np.ndarray]


@dataclass
class Dataset:
    B_obs: np.ndarray
    P: np.ndarray
    H: np.ndarray
    Bs: List[np.ndarray]
    ix2target: dict

    def __post_init__(self):
        # === OTHER MATRICES FOR OBSERVATIONAL SETTING ===
        self.C_obs = self.B_obs @ self.P @ self.H
        self.Theta_obs = self.C_obs.T @ self.C_obs

        # === COMPUTE THETA_k MATRICES ===
        self.Thetas = []
        for B in self.Bs:
            assert B.shape == self.B_obs.shape
            C = B @ self.P @ self.H
            Theta = C.T @ C
            self.Thetas.append(Theta)

    def sample_thetas(self, nsamples):
        mv = multivariate_normal(cov=inv(self.Theta_obs))
        samples = mv.rvs(nsamples)
        Theta_obs = inv(np.cov(samples, rowvar=False))
        Theta_obs = 1/2 * (Theta_obs + Theta_obs.T)
        
        Thetas = []
        for Theta in self.Thetas:
            mv = multivariate_normal(cov=inv(Theta))
            samples = mv.rvs(nsamples)
            Thetas.append(inv(np.cov(samples, rowvar=False)))
        Thetas = [1/2 * (Theta + Theta.T) for Theta in Thetas]
        obs_ds = ObservedDataset(Theta_obs, Thetas)

        return obs_ds


