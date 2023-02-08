# === IMPORTS: THIRD-PARTY ===
import numpy as np
from numpy.linalg import cholesky, inv, pinv
import causaldag as cd
from tqdm import trange

# === IMPORTS: LOCAL ===
from src.utils.linalg import get_rank_one_factors, normalize, normalize_H, orthogonal_projection_matrix
from src.dataset import Dataset
from src.rank_tester import ExactRankOneScorer, RankOneScorer
from src.row_extractor import SingleRowExtractor, ExactSingleRowExtractor
from src.utils.misc import argmax_dict


class IterativeProjectionPartialOrderSolver:
    def __init__(
        self,
        rank_one_scorer: RankOneScorer = ExactRankOneScorer(),
        single_row_extractor: SingleRowExtractor = ExactSingleRowExtractor(),
        rank_gamma: float = 1 - 1e-8,
        num_latent: int = None
    ):
        self.rank_one_scorer = rank_one_scorer
        self.single_row_extractor = single_row_extractor
        self.rank_gamma = rank_gamma
        self.num_latent = num_latent

    def prune_ancestors(
        self,
        envs2qvecs: dict,
        Theta_node: np.ndarray,
        Theta_obs: np.ndarray,
        ancestor_dict: dict,
        verbose=False
    ):
        parents = set(envs2qvecs.keys())
        for env_ix, _ in envs2qvecs.items():
            other_qvecs = [qvec for e, qvec in envs2qvecs.items() if e != env_ix]
            projectedThetaDiff = orthogonal_projection_matrix(other_qvecs, Theta_node - Theta_obs)
            score = self.rank_one_scorer.score(projectedThetaDiff)
            if score > self.rank_gamma:
                parents.remove(env_ix)
            if verbose: print(score, env_ix)
        
        all_ancestors = parents.copy()
        for p in parents:
            all_ancestors |= ancestor_dict[p]
        qvecs_ancestors = [envs2qvecs[node] for node in all_ancestors]
        projectedThetaDiff = orthogonal_projection_matrix(qvecs_ancestors, Theta_node - Theta_obs)
        
        new_qvec = get_rank_one_factors(projectedThetaDiff)[0]
        new_qvec_norm = normalize(new_qvec)

        return new_qvec_norm, all_ancestors

    def pick_next_node(
        self,
        envs2qvecs: dict,
        remaining_Thetas: dict,
        Theta_obs: np.ndarray,
        ancestor_dict: dict,
        verbose = False
    ):
        full_qvecs = [qvec for _, qvec in envs2qvecs.items()]
        projectedThetaDiffs = {
            env_ix: orthogonal_projection_matrix(full_qvecs, Theta - Theta_obs)
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
            ancestor_dict,
            verbose=verbose
        )
        ancestor_dict[next_env_ix] = ancestors

        if verbose:
            print(f"Picked {next_env_ix} with ancestors {ancestors}")

        return next_env_ix, ancestors, qvec

    def solveQ(
        self,
        ds: Dataset, 
        true_ds=None, 
        verbose=False
    ) -> np.ndarray:
        Theta_obs = ds.Theta_obs
        p = Theta_obs.shape[0]
        d = self.num_latent
        remaining_Thetas = dict(enumerate(ds.Thetas))
        envs2qvecs = dict()

        partial_order = cd.DAG(nodes=set(range(d)))
        env_ix2target = dict()
        ancestor_dict = dict()
        for latent_node_ix in trange(d-1, -1, -1):
            next_env_ix, parent_env_ixs, new_qvec = self.pick_next_node(
                envs2qvecs,
                remaining_Thetas,
                Theta_obs,
                ancestor_dict,
                verbose=verbose
            )
            partial_order.add_arcs_from(
                {(p_ix, next_env_ix) for p_ix in parent_env_ixs}
            )

            env_ix2target[next_env_ix] = latent_node_ix
            envs2qvecs[next_env_ix] = new_qvec
            del remaining_Thetas[next_env_ix]

        # R, Q = RQ_partial_order(ds.H, partial_order)
        Q_est = np.zeros((d, p))
        for env_ix, qvec in envs2qvecs.items():
            Q_est[env_ix2target[env_ix]] = qvec

        return partial_order, Q_est, env_ix2target

    def solve_given_Q(
        self,
        ds: Dataset,
        Q_est: np.ndarray,
        env_ix2target
    ):
        Q_est_inv = pinv(Q_est)
        orthogonalized_Theta0 = Q_est_inv.T @ ds.Theta_obs @ Q_est_inv
        C0_est = cholesky(orthogonalized_Theta0).T  # L such that L L.T == M
        orthogonalized_Thetas = {
            ix: Q_est_inv.T @ Theta @ Q_est_inv
            for ix, Theta in enumerate(ds.Thetas)
        }
        C_ests = {ix: cholesky(ot).T for ix, ot in orthogonalized_Thetas.items()}

        R_est = np.zeros((self.num_latent, self.num_latent))
        ix2target = dict()
        for ix, C_est in C_ests.items():
            diff = C_est - C0_est
            target, v1 = self.single_row_extractor.extract_row(diff, forbidden_rows=ix2target.values())
            ix2target[ix] = target
            v2 = C0_est[target]
            rvec = v1 + v2
            R_est[target] = rvec

        H_est = R_est @ Q_est
        H_est, lambda_vals = normalize_H(H_est, return_scaling=True)
        H_est_inv = pinv(H_est)
        B0_est = cholesky(H_est_inv.T @ ds.Theta_obs @ H_est_inv).T
        B_ests = dict()
        for ix in range(len(ds.Thetas)):
            B_est = B0_est.copy()
            B_est[ix2target[ix]] = 0
            lam = abs(lambda_vals[ix2target[ix]])
            B_est[ix2target[ix], ix2target[ix]] = lam
            B_ests[ix] = B_est
        
        # B_ests2 = {
        #     ix: cholesky(H_est_inv.T @ Theta @ H_est_inv).T
        #     for ix, Theta in enumerate(ds.Thetas)
        # }
        # for ix in B_ests:
        #     print(np.max(np.abs(B_ests[ix] - B_ests2[ix])))
        # breakpoint()

        sol = dict(
            H_est=H_est,
            B0_est=B0_est,
            B_ests=B_ests,
            Q_est=Q_est,
            ix2target=ix2target
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


