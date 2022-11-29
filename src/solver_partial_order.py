# === IMPORTS: THIRD-PARTY ===
import numpy as np
from numpy.linalg import cholesky, inv
import causaldag as cd

# === IMPORTS: LOCAL ===
from src.utils.linalg import SubspaceFloat, get_rank_one_factors, normalize, normalize_H
from src.dataset import Dataset
from src.rank_tester import ExactRankOneScorer, RankOneScorer, SVDRankOneScorer
from src.row_extractor import SingleRowExtractor, MaxNormSingleRowExtractor
from src.utils.misc import argmax_dict


class IterativeProjectionPartialOrderSolver:
    def __init__(
        self,
        rank_one_scorer: RankOneScorer = ExactRankOneScorer(),
        single_row_extractor: SingleRowExtractor = MaxNormSingleRowExtractor(),
        rank_gamma: float = 1 - 1e-8
    ):
        self.rank_one_scorer = rank_one_scorer
        self.single_row_extractor = single_row_extractor
        self.rank_gamma = rank_gamma

    def prune_ancestors(
        self,
        envs2qvecs: dict,
        Theta_node: np.ndarray,
        Theta_obs: np.ndarray,
    ):
        ancestors = set(envs2qvecs.keys())
        for env_ix, _ in envs2qvecs.items():
            other_qvecs = [qvec for e, qvec in envs2qvecs.items() if e != env_ix]
            subspace = SubspaceFloat(other_qvecs, vlength=Theta_obs.shape[0])
            projectedThetaDiff = subspace.project_orth_orth(Theta_node - Theta_obs)
            score = self.rank_one_scorer.score(projectedThetaDiff)
            if score > self.rank_gamma:
                ancestors.remove(env_ix)

        qvecs_ancestors = [envs2qvecs[node] for node in ancestors]
        subspace = SubspaceFloat(qvecs_ancestors, vlength=Theta_obs.shape[0])
        projectedThetaDiff = subspace.project_orth_orth(Theta_node - Theta_obs)
        new_qvec = get_rank_one_factors(projectedThetaDiff)[0]

        return new_qvec, ancestors

    def pick_next_node(
        self,
        envs2qvecs: dict,
        remaining_Thetas: dict,
        Theta_obs: np.ndarray
    ):
        full_qvecs = [qvec for _, qvec in envs2qvecs.items()]
        full_subspace = SubspaceFloat(full_qvecs, vlength=Theta_obs.shape[0])
        projectedThetaDiffs = {
            env_ix: full_subspace.project_orth_orth(Theta - Theta_obs)
            for env_ix, Theta in remaining_Thetas.items()
        }
        rank_one_scores = {
            env_ix: self.rank_one_scorer.score(pdiff)
            for env_ix, pdiff in projectedThetaDiffs.items()
        }
        
        rank_one_Theta_env_ixs = argmax_dict(rank_one_scores, one_choice=True)
        next_env_ix = list(rank_one_Theta_env_ixs)[0]
        Theta = remaining_Thetas[next_env_ix]

        qvec, ancestors = self.prune_ancestors(
            envs2qvecs,
            Theta,
            Theta_obs,
        )

        return next_env_ix, ancestors, qvec

    def solveQ(
        self,
        ds: Dataset, 
        true_ds=None, 
        verbose=False
    ) -> np.ndarray:
        Theta_obs = ds.Theta_obs
        p = Theta_obs.shape[0]
        remaining_Thetas = dict(enumerate(ds.Thetas))
        envs2qvecs = dict()

        partial_order = cd.DAG()
        env_ix2target = dict()
        for latent_node_ix in range(p-1, -1, -1):
            next_env_ix, parent_env_ixs, new_qvec = self.pick_next_node(
                envs2qvecs, 
                remaining_Thetas, 
                Theta_obs, 
            )
            partial_order.add_arcs_from(
                {(env_ix2target[p_ix], latent_node_ix) for p_ix in parent_env_ixs}
            )

            env_ix2target[next_env_ix] = latent_node_ix
            envs2qvecs[next_env_ix] = normalize(new_qvec)
            del remaining_Thetas[next_env_ix]

        # R, Q = RQ_partial_order(ds.H, partial_order)
        Q_est = np.zeros((p, p))
        for env_ix, qvec in envs2qvecs.items():
            Q_est[env_ix2target[env_ix]] = qvec

        return partial_order, Q_est, env_ix2target

    def solve_given_Q(
        self,
        ds: Dataset,
        Q_est: np.ndarray,
        env_ix2target
    ):
        Q_est_inv = inv(Q_est)
        orthogonalized_Theta0 = Q_est_inv.T @ ds.Theta_obs @ Q_est_inv
        C0_est = cholesky(orthogonalized_Theta0).T  # L such that L L.T == M
        orthogonalized_Thetas = {
            ix: Q_est_inv.T @ Theta @ Q_est_inv
            for ix, Theta in enumerate(ds.Thetas)
        }
        C_ests = {ix: cholesky(ot).T for ix, ot in orthogonalized_Thetas.items()}

        R_est = np.zeros(Q_est.shape)
        ix2target = dict()
        for ix, C_est in C_ests.items():
            diff = C_est - C0_est
            target, v1 = self.single_row_extractor.extract_row(diff, forbidden_rows=ix2target.values())
            ix2target[ix] = target
            v2 = C0_est[target]
            rvec = v1 + v2
            R_est[target] = rvec

        H_est = R_est @ Q_est
        H_est = normalize_H(H_est)
        H_est_inv = inv(H_est)
        B0_est = cholesky(H_est_inv.T @ ds.Theta_obs @ H_est_inv).T
        B_ests = {
            ix: cholesky(H_est_inv.T @ Theta @ H_est_inv).T
            for ix, Theta in enumerate(ds.Thetas)
        }

        sol = dict(
            H_est=H_est,
            B0_est=B0_est,
            B_ests=B_ests,
            Q_est=Q_est,
            ix2target=env_ix2target
        )
        return sol

    def solve(
        self, 
        ds: Dataset, 
        true_ds=None,
        verbose=False
    ):
        partial_order, Q_est, env_ix2target = self.solveQ(ds, true_ds=true_ds, verbose=verbose)
        sol = self.solve_given_Q(ds, Q_est, env_ix2target)
        sol["partial_order"] = partial_order
        return sol

    def check_solution(
        self,
        sol,
        ds: Dataset
    ):
        # === STEP 1: CHECK THAT THE SOLUTION GIVES THE RIGHT OUTPUT
        H_est = sol["H_est"]
        B0_est = sol["B0_est"]
        B_ests = sol["B_ests"]
        ix2target = sol["ix2target"]
        Theta_obs_est = H_est.T @ B0_est.T @ B0_est @ H_est
        Theta_ests = {ix: H_est.T @ B.T @ B @ H_est for ix, B in B_ests.items()}
        
        matches_observational = np.allclose(Theta_obs_est, ds.Theta_obs)
        matches_interventional = all([np.allclose(Theta_est, ds.Thetas[ix]) for ix, Theta_est in Theta_ests.items()])

        # === STEP 2: CHECK THAT INTERVENTIONS CORRESPOND TO HARD INTERVENTIONS
        all_interventions_perfect = True
        for ix, target in ix2target.items():
            B_est = B_ests[ix]
            parent_weights = B_est[target][(target+1):]
            is_zero = np.allclose(parent_weights, 0)
            all_interventions_perfect &= is_zero

        return (
            matches_observational
            &
            matches_interventional
            &
            all_interventions_perfect
        )
