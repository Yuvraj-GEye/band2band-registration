import numpy as np
import torch


from galaxeye_fuse.logger.mlflow_instance import get_aim_run
from galaxeye_fuse.logger.mlflow_log import log_with_run

from sklearn.utils._random import sample_without_replacement

from functools import partial


class RANSACRegressorGPU:
    def __init__(
        self,
        estimator=None,
        *,
        min_samples=None,
        residual_threshold=None,
        max_trials=100,
        max_skips=np.inf,
        stop_n_inliers=np.inf,
        stop_score=np.inf,
        stop_probability=0.99,
        loss="absolute_error",
        random_state=None,
        device=None,
        dtype=torch.float32,
        batch_size=128,
    ):
        self.estimator = estimator
        self.min_samples = min_samples
        self.residual_threshold = residual_threshold
        self.max_trials = max_trials
        self.max_skips = max_skips
        self.stop_n_inliers = stop_n_inliers
        self.stop_score = stop_score
        self.stop_probability = stop_probability
        self.loss = loss
        self.random_state = random_state
        self.dtype = dtype
        self.device = (
            device
            if device is not None
            else "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )
        self.batch_size = batch_size

        torch.manual_seed(self.random_state)
        if "cuda" in self.device:
            torch.cuda.manual_seed(self.random_state)
            torch.cuda.manual_seed_all(self.random_state)
        self.rng_ = np.random.RandomState(self.random_state)

    def _convert_to_tensor(self, X, move_to_device=True):
        if torch.is_tensor(X):
            if X.dtype != self.dtype or (
                move_to_device and X.device.type != self.device.split(":")[0]
            ):
                return X.to(dtype=self.dtype, device=self.device)
            return X

        if hasattr(X, "toarray"):
            X_array = X.toarray()
            return torch.tensor(
                X_array,
                dtype=self.dtype,
                device=self.device if move_to_device else "cpu",
            )

        return torch.tensor(
            X, dtype=self.dtype, device=self.device if move_to_device else "cpu"
        )

    @log_with_run(run_getter=get_aim_run)
    def fit(self, X, y):
        print(f"runnning optimised version for ransac on {self.device=}")
        torch.backends.cuda.matmul.allow_tf32 = False

        # torch.use_deterministic_algorithms(True)

        torch.cuda.empty_cache()

        X_torch = self._convert_to_tensor(X)
        y_torch = self._convert_to_tensor(y)
        n_samples, n_features = X_torch.shape

        if self.min_samples is None:
            min_samples = n_features + 1
        elif 0 < self.min_samples < 1:
            min_samples = int(np.ceil(self.min_samples * n_samples))
        else:
            min_samples = self.min_samples
        if min_samples > n_samples:
            raise ValueError(f"min_samples ({min_samples}) > n_samples ({n_samples})")

        if self.residual_threshold is None:
            residual_threshold = torch.median(
                torch.abs(y_torch - torch.median(y_torch, dim=0).values)
            ).item()
        else:
            residual_threshold = self.residual_threshold

        ones_full = torch.ones((n_samples, 1), dtype=self.dtype, device=self.device)
        X_full_aug = torch.cat([X_torch, ones_full], dim=1)

        overall_best_inlier_count = -1

        overall_best_mask = None

        # Optionally partial the function for fixed values
        sample_fn = partial(
            sample_without_replacement,
            n_population=n_samples,
            n_samples=min_samples,
            random_state=self.rng_,
        )

        num_batches = int(np.ceil(self.max_trials / self.batch_size))

        for batch_idx in range(num_batches):
            current_batch = min(
                self.batch_size, self.max_trials - batch_idx * self.batch_size
            )

            candidate_indices = torch.stack(
                [
                    torch.tensor(sample_fn(), device=self.device, dtype=torch.int64)
                    for _ in range(current_batch)
                ],
                dim=0,
            )

            X_candidates = X_torch[candidate_indices]
            y_candidates = y_torch[candidate_indices]

            ones_candidates = torch.ones(
                (current_batch, min_samples, 1), dtype=self.dtype, device=self.device
            )
            X_candidates_aug = torch.cat([X_candidates, ones_candidates], dim=2)
            #################################################### Linear Regressor
            A = torch.bmm(X_candidates_aug.transpose(1, 2), X_candidates_aug)
            reg = 1e-10 * torch.eye(
                n_features + 1, dtype=self.dtype, device=self.device
            ).unsqueeze(0)
            A_reg = A + reg
            B_mat = torch.bmm(X_candidates_aug.transpose(1, 2), y_candidates)
            candidate_solutions = torch.linalg.solve(A_reg, B_mat)

            X_full_aug_exp = X_full_aug.unsqueeze(0).expand(current_batch, -1, -1)
            preds = torch.matmul(X_full_aug_exp, candidate_solutions)

            #######################################################
            if self.loss == "absolute_error":
                residuals = torch.sum(
                    torch.abs(
                        preds - y_torch.unsqueeze(0).expand(current_batch, -1, -1)
                    ),
                    dim=2,
                )

            elif self.loss == "squared_error":
                residuals = torch.sum(
                    (preds - y_torch.unsqueeze(0).expand(current_batch, -1, -1)) ** 2,
                    dim=2,
                )
            else:
                raise ValueError(f"Unsupported loss type: {self.loss}")

            inlier_mask = residuals <= residual_threshold
            inlier_counts = inlier_mask.sum(dim=1)

            batch_best_count, batch_best_idx = torch.max(inlier_counts, dim=0)
            if batch_best_count > overall_best_inlier_count:
                overall_best_inlier_count = batch_best_count.item()

                overall_best_mask = inlier_mask[batch_best_idx]

        if overall_best_mask is None or overall_best_mask.sum() < min_samples:
            raise ValueError("RANSAC failed to find a valid consensus set")

        self.n_trials_ = self.max_trials
        self.inlier_mask_ = overall_best_mask.cpu().numpy()

        return self
