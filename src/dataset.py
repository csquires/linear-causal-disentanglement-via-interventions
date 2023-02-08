# === IMPORTS: BUILT-IN ===
from dataclasses import dataclass
from typing import Dict, Any, List

# === IMPORTS: THIRD-PARTY ===
import numpy as np
from numpy.linalg import inv, pinv
from scipy.stats import multivariate_normal, invwishart


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
        self.precision_latent = self.B_obs.T @ self.B_obs
        self.Theta_obs = self.C_obs.T @ self.C_obs

        # === COMPUTE THETA_k MATRICES ===
        self.Thetas = []
        self.precisions_latent = []
        for B in self.Bs:
            assert B.shape == self.B_obs.shape
            C = B @ self.P @ self.H
            Theta = C.T @ C
            self.Thetas.append(Theta)
            self.precisions_latent.append(B.T @ B)

    def sample_thetas(self, nsamples):
        Theta_latent_obs = invwishart(nsamples, self.precision_latent).rvs(1) * nsamples
        Theta_obs = self.H.T @ Theta_latent_obs @ self.H

        Thetas = []
        for precision_latent in self.precisions_latent:
            Theta_latent = invwishart(nsamples, precision_latent).rvs(1) * nsamples
            Thetas.append(self.H.T @ Theta_latent @ self.H)
        # Thetas = [1/2 * (Theta + Theta.T) for Theta in Thetas]
        obs_ds = ObservedDataset(Theta_obs, Thetas)

        return obs_ds

    def sample(self, nsamples):
        mv = multivariate_normal(cov=pinv(self.Theta_obs), allow_singular=True)
        samples = mv.rvs(nsamples)
        
        interventional_samples = []
        for Theta in self.Thetas:
            mv = multivariate_normal(cov=pinv(Theta), allow_singular=True)
            samples = mv.rvs(nsamples)
            interventional_samples.append(samples)

        return samples, interventional_samples


