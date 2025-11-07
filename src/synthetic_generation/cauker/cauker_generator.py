import functools
import random

import cupy as cp
import networkx as nx
import numpy as np
from sklearn.gaussian_process.kernels import (
    RBF,
    ConstantKernel,
    DotProduct,
    ExpSineSquared,
    RationalQuadratic,
    WhiteKernel,
)
from src.synthetic_generation.abstract_classes import AbstractTimeSeriesGenerator
from src.synthetic_generation.generator_params import CauKerGeneratorParams


class CauKerGenerator(AbstractTimeSeriesGenerator):
    """Structural‑Causal‑Model GP-based time series generator.

    This class is a refactor of the original script-level implementation, exposing
    the same logic as instance methods and generating one multivariate series per call.
    """

    def __init__(self, params: CauKerGeneratorParams):
        self.params = params

    # -------------------------------------------------------------------------
    # 1. Kernel Bank Construction (parameterised by `time_length`)
    # -------------------------------------------------------------------------
    def build_kernel_bank(self, time_length: int) -> list:
        return [
            # Hourly / sub‑hourly cycles
            ExpSineSquared(periodicity=24 / time_length),
            ExpSineSquared(periodicity=48 / time_length),
            ExpSineSquared(periodicity=96 / time_length),
            # Hourly components embedded in weekly structure
            ExpSineSquared(periodicity=24 * 7 / time_length),
            ExpSineSquared(periodicity=48 * 7 / time_length),
            ExpSineSquared(periodicity=96 * 7 / time_length),
            # Daily / sub‑daily
            ExpSineSquared(periodicity=7 / time_length),
            ExpSineSquared(periodicity=14 / time_length),
            ExpSineSquared(periodicity=30 / time_length),
            ExpSineSquared(periodicity=60 / time_length),
            ExpSineSquared(periodicity=365 / time_length),
            ExpSineSquared(periodicity=365 * 2 / time_length),
            # Weekly / monthly / quarterly variations
            ExpSineSquared(periodicity=4 / time_length),
            ExpSineSquared(periodicity=26 / time_length),
            ExpSineSquared(periodicity=52 / time_length),
            ExpSineSquared(periodicity=4 / time_length),
            ExpSineSquared(periodicity=6 / time_length),
            ExpSineSquared(periodicity=12 / time_length),
            ExpSineSquared(periodicity=4 / time_length),
            ExpSineSquared(periodicity=(4 * 10) / time_length),
            ExpSineSquared(periodicity=10 / time_length),
            # Stationary + noise kernels
            DotProduct(sigma_0=0.0),
            DotProduct(sigma_0=1.0),
            DotProduct(sigma_0=10.0),
            RBF(length_scale=0.1),
            RBF(length_scale=1.0),
            RBF(length_scale=10.0),
            RationalQuadratic(alpha=0.1),
            RationalQuadratic(alpha=1.0),
            RationalQuadratic(alpha=10.0),
            WhiteKernel(noise_level=0.1),
            WhiteKernel(noise_level=1.0),
            ConstantKernel(),
        ]

    # -------------------------------------------------------------------------
    # 2. Binary map utility for kernel algebra
    # -------------------------------------------------------------------------
    def random_binary_map(self, a, b):
        binary_ops = [lambda x, y: x + y, lambda x, y: x * y]
        return np.random.choice(binary_ops)(a, b)

    # -------------------------------------------------------------------------
    # 3. Mean‑function library
    # -------------------------------------------------------------------------
    def zero_mean(self, x: np.ndarray) -> np.ndarray:
        return np.zeros_like(x)

    def linear_mean(self, x: np.ndarray) -> np.ndarray:
        a = np.random.uniform(-1.0, 1.0)
        b = np.random.uniform(-1.0, 1.0)
        return a * x + b

    def exponential_mean(self, x: np.ndarray) -> np.ndarray:
        a = np.random.uniform(0.5, 1.5)
        b = np.random.uniform(0.5, 1.5)
        return a * np.exp(b * x)

    def anomaly_mean(self, x: np.ndarray) -> np.ndarray:
        m = np.zeros_like(x)
        num_anomalies = np.random.randint(1, 6)
        for _ in range(num_anomalies):
            idx = np.random.randint(0, len(x))
            m[idx] += np.random.uniform(-5.0, 5.0)
        return m

    def random_mean_combination(self, x: np.ndarray) -> np.ndarray:
        mean_functions = [
            self.zero_mean,
            self.linear_mean,
            self.exponential_mean,
            self.anomaly_mean,
        ]
        m1, m2 = np.random.choice(mean_functions, 2, replace=True)
        combine_ops = [lambda u, v: u + v, lambda u, v: u * v]
        return np.random.choice(combine_ops)(m1(x), m2(x))

    # -------------------------------------------------------------------------
    # 4. GPU‑accelerated sampling from the GP prior
    # -------------------------------------------------------------------------
    def sample_from_gp_prior_efficient_gpu(
        self,
        *,
        kernel,
        X: np.ndarray,
        random_seed: int | None = None,
        method: str = "eigh",
        mean_vec: np.ndarray | None = None,
    ) -> np.ndarray:
        if X.ndim == 1:
            X = X[:, None]

        cov_cpu = kernel(X)
        n = X.shape[0]

        mean_vec = np.zeros(n, dtype=np.float64) if mean_vec is None else mean_vec

        cov_gpu = cp.asarray(cov_cpu)
        mean_gpu = cp.asarray(mean_vec)

        if random_seed is not None:
            cp.random.seed(random_seed)

        ts_gpu = cp.random.multivariate_normal(mean=mean_gpu, cov=cov_gpu, method=method)
        return cp.asnumpy(ts_gpu)

    # -------------------------------------------------------------------------
    # 5. Structural‑Causal‑Model time‑series generator (parameterised)
    # -------------------------------------------------------------------------
    def generate_random_dag(self, num_nodes: int, max_parents: int = 3) -> nx.DiGraph:
        G = nx.DiGraph()
        nodes = list(range(num_nodes))
        random.shuffle(nodes)
        G.add_nodes_from(nodes)
        for i in range(num_nodes):
            possible_parents = nodes[:i]
            num_par = np.random.randint(0, min(len(possible_parents), max_parents) + 1)
            for p in random.sample(possible_parents, num_par):
                G.add_edge(p, nodes[i])
        return G

    def random_activation(self, x: np.ndarray, func_type: str = "linear") -> np.ndarray:
        if func_type == "linear":
            a = np.random.uniform(0.5, 2.0)
            b = np.random.uniform(-1.0, 1.0)
            return a * x + b
        if func_type == "relu":
            return np.maximum(0.0, x)
        if func_type == "sigmoid":
            return 1.0 / (1.0 + np.exp(-x))
        if func_type == "sin":
            return np.sin(x)
        if func_type == "mod":
            c = np.random.uniform(1.0, 5.0)
            return np.mod(x, c)
        # default: leaky‑ReLU
        alpha = np.random.uniform(0.01, 0.3)
        return np.where(x > 0, x, alpha * x)

    def random_edge_mapping(self, parents_data: list[np.ndarray]) -> np.ndarray:
        combined = np.stack(parents_data, axis=1)
        W = np.random.randn(len(parents_data))
        b = np.random.randn()
        non_linear_input = combined @ W + b
        chosen_func = np.random.choice(["linear", "relu", "sigmoid", "sin", "mod", "leakyrelu"])
        return self.random_activation(non_linear_input, chosen_func)

    # -------------------------------------------------------------------------
    # 6. End‑to‑end SCM sampler
    # -------------------------------------------------------------------------
    def generate_scm_time_series(
        self,
        *,
        time_length: int,
        num_features: int,
        max_parents: int,
        seed: int,
        num_nodes: int,
    ) -> dict[int, np.ndarray]:
        np.random.seed(seed)
        random.seed(seed)

        dag = self.generate_random_dag(num_nodes, max_parents=max_parents)
        kernel_bank = self.build_kernel_bank(time_length)

        root_nodes = [n for n in dag.nodes if dag.in_degree(n) == 0]
        node_data: dict[int, np.ndarray] = {}

        X = np.linspace(0.0, 1.0, time_length)

        # Sample roots directly from the GP prior
        for r in root_nodes:
            selected_kernels = np.random.choice(kernel_bank, np.random.randint(1, 8), replace=True)
            kernel = functools.reduce(self.random_binary_map, selected_kernels)
            mean_vec = self.random_mean_combination(X)
            node_data[r] = self.sample_from_gp_prior_efficient_gpu(
                kernel=kernel, X=X, mean_vec=mean_vec, random_seed=seed
            )

        # Propagate through DAG
        for node in nx.topological_sort(dag):
            if node in root_nodes:
                continue
            parents = list(dag.predecessors(node))
            parents_ts = [node_data[p] for p in parents]
            node_data[node] = self.random_edge_mapping(parents_ts)

        return node_data

    # -------------------------------------------------------------------------
    # Public API: generate one multivariate series (length, num_channels)
    # -------------------------------------------------------------------------
    def generate_time_series(self, random_seed: int | None = None) -> np.ndarray:
        """Generate one multivariate series with shape (length, num_channels)."""
        seed = self.params.global_seed if random_seed is None else random_seed

        # Resolve num_channels which can be int or (min, max)
        desired_channels: int | tuple[int, int] = self.params.num_channels
        if isinstance(desired_channels, tuple):
            low, high = desired_channels
            if low > high:
                low, high = high, low
            num_channels = int(np.random.default_rng(seed).integers(low, high + 1))
        else:
            num_channels = int(desired_channels)

        if num_channels > self.params.num_nodes:
            raise ValueError(f"num_channels ({num_channels}) cannot exceed num_nodes ({self.params.num_nodes}).")

        node_data = self.generate_scm_time_series(
            time_length=self.params.length,
            num_features=num_channels,
            max_parents=self.params.max_parents,
            seed=seed,
            num_nodes=self.params.num_nodes,
        )

        chosen_nodes = random.sample(list(node_data.keys()), num_channels)
        channels = [node_data[n].astype(np.float32) for n in chosen_nodes]
        values = np.stack(channels, axis=1)  # (length, num_channels)
        return values
