from abc import ABC, abstractmethod
from collections import defaultdict

import numpy as np


class GradientVariance(ABC):
    """Base class for gradient variance computations under different
    parameter initialization distributions.

    Subclasses must implement the three moment primitives:
      - Egsq_diag(idx): E[g_idx^2] diagonal (single-path) contribution
      - Egsq_mix(idx): E[g_idx^2] cross-path mixing contribution
      - Eg_sq(idx): E[g_idx]^2, the squared mean
    """

    @abstractmethod
    def Egsq_diag(self, idx: int) -> float:
        raise NotImplementedError

    @abstractmethod
    def Egsq_mix(self, idx: int) -> float:
        raise NotImplementedError

    @abstractmethod
    def Eg_sq(self, idx: int) -> float:
        raise NotImplementedError

    def per_parameter(self, idx: int) -> float:
        """Var(g_idx) = E[g_idx^2] - E[g_idx]^2, where
        E[g_idx^2] = Egsq_diag(idx) + Egsq_mix(idx)."""
        return self.Egsq_diag(idx) + self.Egsq_mix(idx) - self.Eg_sq(idx)

    def average(self) -> dict:
        """Average of each underlying term across all parameters, plus the
        average variance."""
        diag = np.mean([self.Egsq_diag(idx) for idx in range(self.num_params)])
        mix = np.mean([self.Egsq_mix(idx) for idx in range(self.num_params)])
        sq = np.mean([self.Eg_sq(idx) for idx in range(self.num_params)])
        return {
            "Egsq_diag": diag,
            "Egsq_mix": mix,
            "Eg_sq": sq,
            "all": diag + mix - sq,
        }

    def sampled(self, num_samples: int = 500):
            rng = np.random.default_rng()
            all_grads = np.empty((num_samples, self.num_params))
            for i in range(num_samples):
                theta = getattr(rng, self.random)(*self.random_args, size=self.num_params)
                _, grads = self.eval_and_grad(theta)
                all_grads[i] = grads[0]
            var_per_param = np.var(all_grads, axis=0)
            return {
                "mean": float(np.mean(var_per_param)),
                "std": float(np.std(var_per_param)),
        }


class Uniform(GradientVariance):
    def __init__(self, prop, expr_idx: int = 0):
        self.expression = prop.exprs[expr_idx]
        self.num_params = prop.num_params
        self.eval_and_grad = prop.eval_and_grad
        self.random = "uniform"
        self.random_args = (0, 2 * np.pi)
        # TODO: implement range different than [0, 2*np.pi]

    def Egsq_diag(self, idx: int) -> float:
        var = 0.0
        for coeff, sin_idx, cos_idx in self.expression:
            S_x = list(sin_idx) + list(cos_idx)
            if idx in S_x:
                var += coeff**2 * 2**(-len(S_x))
        return var

    def Egsq_mix(self, idx: int) -> float:
        return 0

    def Eg_sq(self, idx: int) -> float:
        return 0


class Gaussian(GradientVariance):
    def __init__(self, prop, sigma: float, expr_idx: int = 0):
        self.expression = prop.exprs[expr_idx]
        self.sigma = sigma
        self.num_params = prop.num_params
        self.eval_and_grad = prop.eval_and_grad
        self.random = "normal"
        self.random_args = (0, sigma)   # loc=0, scale=sigma

        self.mu_s = (1 - np.exp(-2 * sigma**2)) / 2
        self.mu_c = (1 + np.exp(-2 * sigma**2)) / 2
        self.e_cos = np.exp(-sigma**2 / 2)
        self.sin_sets = [set(s) for _, s, _ in self.expression]
        self.cos_sets = [set(c) for _, _, c in self.expression]
        self.coeffs = [c for c, _, _ in self.expression]

        self._mix_cache = None  # computed lazily, see Egsq_mix

    def Egsq_diag(self, idx: int) -> float:
        var = 0.0
        for coeff, sin_idx, cos_idx in self.expression:
            ns, nc = len(sin_idx), len(cos_idx)
            if idx in sin_idx:
                var += coeff**2 * self.mu_s**(ns - 1) * self.mu_c**(nc + 1)
            elif idx in cos_idx:
                var += coeff**2 * self.mu_s**(ns + 1) * self.mu_c**(nc - 1)
        return var

    def Egsq_mix(self, idx: int) -> float:
        if self._mix_cache is None:
            self._mix_cache = self._compute_all_mix()
        return self._mix_cache[idx]

    def Eg_sq(self, idx: int) -> float:
        total = 0.0
        for coeff, sin_idx, cos_idx in self.expression:
            if len(sin_idx) == 1 and sin_idx[0] == idx:
                s = 1 + len(cos_idx)
                total += coeff * self.e_cos**s
        return total**2

    def average(self) -> dict:
        mu_s = (1 - np.exp(-2 * self.sigma**2)) / 2
        mu_c = (1 + np.exp(-2 * self.sigma**2)) / 2
        e_cos = np.exp(-self.sigma**2 / 2)

        # precompute sets once
        sin_sets = [set(s) for _, s, _ in self.expression]
        cos_sets = [set(c) for _, _, c in self.expression]
        coeffs   = [c for c, _, _ in self.expression]
        n        = len(self.expression)

        # E[g^2] diagonal terms
        gq_non_mixed = 0.0
        for i, (coeff, sin_idx, cos_idx) in enumerate(self.expression):
            ns = len(sin_idx)
            nc = len(cos_idx)
            term_sin = ns * mu_s**(ns - 1) * mu_c**(nc + 1) if ns > 0 else 0.0
            term_cos = nc * mu_s**(ns + 1) * mu_c**(nc - 1) if nc > 0 else 0.0
            gq_non_mixed += coeff**2 * (term_sin + term_cos)

        # E[g^2] mixed terms
        # group by frozenset of sin indices for fast filtering
        sin_groups = defaultdict(list)
        for idx in range(n):
            sin_groups[frozenset(sin_sets[idx])].append(idx)

        gq_mixed = 0.0
        for idx_x in range(n):
            sx, cx = sin_sets[idx_x], cos_sets[idx_x]
            all_x  = sx | cx

            for idx_y in range(idx_x + 1, n):
                sy, cy = sin_sets[idx_y], cos_sets[idx_y]

                # unshared sin filter: cheapest check first
                if sx - sy - cx or sy - sx - cy:
                    continue

                all_y    = sy | cy
                shared   = all_x & all_y
                unshared = all_x.symmetric_difference(all_y)
                unshared_factor = e_cos ** len(unshared)

                # precompute shared type agreement and product once
                # check condition 2 for all shared indices at once
                shared_types_agree = all((i in sx) == (i in sy) for i in shared)

                for k in shared:
                    k_type_x = 's' if k in sx else 'c'
                    k_type_y = 's' if k in sy else 'c'

                    if k_type_x != k_type_y:
                        continue

                    # condition 2: check shared - {k}
                    # if all shared types agree, no need to recheck per k
                    if not shared_types_agree:
                        if any((i in sx) != (i in sy) for i in shared - {k}):
                            continue

                    M_k = mu_c if k_type_x == 's' else mu_s

                    shared_prod = 1.0
                    for i in shared - {k}:
                        shared_prod *= mu_s if i in sx else mu_c

                    gq_mixed += 2 * coeffs[idx_x] * coeffs[idx_y] * M_k * shared_prod * unshared_factor

        # E[g]^2 term: only paths with ns == 1 contribute
        paths_by_k = defaultdict(list)
        for coeff, sin_idx, cos_idx in self.expression:
            if len(sin_idx) == 1:
                k = sin_idx[0]
                s = 1 + len(cos_idx)  # |S_x| = ns + nc = 1 + nc
                paths_by_k[k].append(coeff * e_cos**s)  # fix 2: store c_x * e^{-σ²|S_x|/2}

        mean_sq = sum(s**2 for s in (sum(v) for v in paths_by_k.values()))  # square the inner sum

        return {
            "Egsq_diag": gq_non_mixed / self.num_params,
            "Egsq_mix": gq_mixed / self.num_params,
            "Eg_sq": mean_sq / self.num_params,
            "all": (gq_non_mixed + gq_mixed - mean_sq) / self.num_params,
        }

    def get_peak(self):
        dominant_lengths = [
            len(cos)
            for _, sin, cos in self.expression
            if len(sin) == 0
        ]
        n = np.median(dominant_lengths)  # or mode, or mean
        sigma_opt = np.sqrt(0.5 * np.log(n / (n - 2)))
        return sigma_opt