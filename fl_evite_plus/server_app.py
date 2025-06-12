# fl_evite_plus/server_app.py
"""Hierarchical FL server (distances embedded in mst.json) using FedProx as the base strategy."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional

import os
import numpy as np
import ray
import warnings

from sklearn.metrics import log_loss, accuracy_score

from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.client_manager import ClientProxy
# ─── Replace FedAvg import with FedProx ────────────────────────────────────────
from flwr.server.strategy import FedProx, Strategy
from flwr.server.strategy.aggregate import aggregate
from flwr.common import (
    Context,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
    Parameters,
    FitRes,
    EvaluateRes,
    FitIns,
    EvaluateIns,
    Scalar,
)

from fl_evite_plus.task import get_model, set_model_params, load_full_test_data

# --------------------------------------------------------------------------- #
#  Tree utilities                                                             #
# --------------------------------------------------------------------------- #
def load_tree_with_weights(path: str) -> Dict[int, Dict[int, float]]:
    """Load MST JSON as parent → {child: distance} map with int keys."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return {int(p): {int(c): float(d) for c, d in kids.items()} for p, kids in raw.items()}


def all_nodes(weight_tree: Dict[int, Dict[int, float]]) -> Set[int]:
    """Return the full set of nodes (parents ∪ children)."""
    parents = set(weight_tree.keys())
    children = {c for kids in weight_tree.values() for c in kids}
    return parents | children


def param_bits(params: List[np.ndarray]) -> int:
    """Count total bits for a list of arrays (float32 → 32 bits/element)."""
    return int(sum(p.size for p in params) * 32)


# --------------------------------------------------------------------------- #
#  Hierarchical strategy                                                      #
# --------------------------------------------------------------------------- #
class HierarchicalStrategy(Strategy):
    """Bottom‐up aggregation with server‐side energy accounting, wrapping a FedProx base."""

    def __init__(
        self,
        weight_tree: Dict[int, Dict[int, float]],
        base: Strategy,
        energy_per_bit: float,
    ):
        super().__init__()
        self.tree = weight_tree
        self.base = base
        self.e_bit = energy_per_bit
        self.valid_cids = all_nodes(weight_tree) - {0}
        self.cid_to_uuid: Dict[int, str] = {}  # Maps logical cid → Flower UUID

    def initialize_parameters(self, client_manager) -> Optional[Parameters]:
        return self.base.initialize_parameters(client_manager)

    def configure_fit(
        self,
        server_round: int,
        parameters: Parameters,
        client_manager,
    ) -> List[Tuple[ClientProxy, FitIns]]:
        all_clients = client_manager.all()  # Dict[str, ClientProxy]

        if server_round == 1:
            # Round 1: allow all clients
            fit_ins_list = [
                (c, FitIns(parameters, {"server_round": server_round}))
                for c in all_clients.values()
            ]
            return fit_ins_list

        # Round >1: select only clients whose cid is in the MST
        selected = []
        for cid, uuid in self.cid_to_uuid.items():
            if cid in self.valid_cids and uuid in all_clients:
                selected.append((all_clients[uuid], FitIns(parameters, {"server_round": server_round})))
        return selected

    def configure_evaluate(
        self,
        server_round: int,
        parameters: Parameters,
        client_manager,
    ) -> List[Tuple[ClientProxy, EvaluateIns]]:
        all_clients = client_manager.all()

        if server_round == 1:
            return [
                (c, EvaluateIns(parameters, {}))
                for c in all_clients.values()
            ]

        selected = []
        for cid, uuid in self.cid_to_uuid.items():
            if cid in self.valid_cids and uuid in all_clients:
                selected.append((all_clients[uuid], EvaluateIns(parameters, {})))
        return selected

    def _cid_updates(
        self, results: List[Tuple[ClientProxy, FitRes]]
    ) -> Dict[int, Tuple[List[np.ndarray], int]]:
        mapping: Dict[int, Tuple[List[np.ndarray], int]] = {}
        for client_proxy, fit_res in results:
            if isinstance(fit_res, ray.ObjectRef):
                fit_res = ray.get(fit_res)
            params = fit_res.parameters
            if isinstance(params, ray.ObjectRef):
                params = ray.get(params)

            nds = parameters_to_ndarrays(params)
            n_examples = fit_res.num_examples
            cid = fit_res.metrics.get("cid")
            if cid is None:
                raise ValueError("Missing 'cid' in FitRes.metrics")

            self.cid_to_uuid[int(cid)] = client_proxy.cid
            mapping[int(cid)] = (nds, n_examples)
        return mapping

    def _aggregate_subtree(
        self,
        node: int,
        parent: Optional[int],
        updates: Dict[int, Tuple[List[np.ndarray], int]],
    ) -> Tuple[List[np.ndarray], int, float]:
        triplets: List[Tuple[List[np.ndarray], int, float]] = []

        # Recurse into children
        for child in self.tree.get(node, {}):
            #child_params, child_n, child_e = self._aggregate_subtree(child, node, updates)
            try:
                child_params, child_n, child_e = self._aggregate_subtree(child, node, updates)
            except ValueError:
                # no updates in that branch → skip it
                continue
            triplets.append((child_params, child_n, child_e))

        # If this node itself is a client leaf, include its own update
        if node in updates:
            params, n = updates[node]
            energy = 0.0
            if parent is not None:
                dist = self.tree[parent][node]
                energy = self.e_bit * (dist**2) * param_bits(params) #energy model
            triplets.append((params, n, energy))

        # If after pruning children & no self‐update, bail out
        if not triplets:
            raise ValueError(f"No update for node {node}")
        
        # 1) Compute a weighted FedAvg of all (params, n) in this subtree
        weight_tuples = [(p[0], n) for (p, n, _) in triplets]
        agg_weights = aggregate(weight_tuples)
        total_n = sum(n for (_, n, _) in triplets)

        # 2) If model has intercept, aggregate that too
        if len(triplets[0][0]) > 1:
            intercept_tuples = [(p[1], n) for (p, n, _) in triplets]
            agg_intercept = aggregate(intercept_tuples)
            agg_params = [agg_weights, agg_intercept]
        else:
            agg_params = [agg_weights]

        # 3) Sum up the energy cost from the subtree
        total_energy = sum(e for (_, _, e) in triplets)
        return agg_params, total_n, total_energy

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[BaseException],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        if not results:
            return None, {}

        updates = self._cid_updates(results)
        root_nds, _, total_energy = self._aggregate_subtree(0, None, updates)
        return ndarrays_to_parameters(root_nds), {"round_total_energy": total_energy}

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[BaseException],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        if not results:
            return None, {}

        accs = [
            ev_res.metrics.get("accuracy", 0.0)
            for _, ev_res in results
            if ev_res.metrics
        ]
        avg_acc = float(sum(accs) / len(accs)) if accs else 0.0
        return 0.0, {"round_avg_accuracy": avg_acc}


    def evaluate(
        self,
        server_round: int,
        parameters: Parameters,
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        # 1) Load the full centralized test split
        X_test, y_test = load_full_test_data(num_parts=10)

        # 2) Create a fresh LogisticRegression and initialize classes_/coef_/intercept_
        model = get_model(penalty="l2", local_epochs=1)
        from fl_evite_plus.task import set_initial_params
        set_initial_params(model)   # <--- ADD THIS LINE to set model.classes_

        # 3) Unpack the federated parameters into that model
        ndarrays = parameters_to_ndarrays(parameters)
        set_model_params(model, ndarrays)

        # 4) Make predictions
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)

        # 5) Guard against NaNs/Infs
        if np.isnan(y_proba).any() or np.isinf(y_proba).any():
            warnings.warn("NaN or Inf detected in predictions. Skipping evaluation.")
            return float("nan"), {"centralized_accuracy": 0.0}

        

        loss = log_loss(y_test, y_proba, labels=np.arange(10))
        acc = accuracy_score(y_test, y_pred)

        return float(loss), {"centralized_accuracy": float(acc)}

# --------------------------------------------------------------------------- #
#  ServerApp entry point                                                      #
# --------------------------------------------------------------------------- #
def server_fn(context: Context) -> ServerAppComponents:
    cfg = context.run_config
    # Allow override from environment
    if "MST_FILE" in os.environ:
        cfg["mst_file"] = os.environ["MST_FILE"]

    tree = load_tree_with_weights(cfg["mst_file"])
    num_clients = len(all_nodes(tree))

    # 1) Initialize a fresh LogisticRegression (same as before)
    model = get_model(cfg["penalty"], cfg["local-epochs"])
    # 2) Set zero‐vector initial parameters
    from flwr.common import ndarrays_to_parameters
    from fl_evite_plus.task import set_initial_params, get_model_params

    set_initial_params(model)
    initial_nds = get_model_params(model)
    initial_parameters = ndarrays_to_parameters(initial_nds)

    # 3) Build a FedProx base strategy instead of FedAvg
    mu = cfg.get("fedprox_mu", 0.01)  # Proximal term; tune as needed
    base = FedProx(
        proximal_mu=mu,
        fraction_fit=cfg.get("fraction_fit", 1.0),
        fraction_evaluate=cfg.get("fraction_evaluate", 1.0),
        min_available_clients=num_clients,
        initial_parameters=initial_parameters,
        on_fit_config_fn=lambda rnd: {"server_round": rnd, "mst_file": cfg["mst_file"]},
        on_evaluate_config_fn=lambda rnd: {"server_round": rnd, "mst_file": cfg["mst_file"]},
    )

    strategy = HierarchicalStrategy(
        weight_tree=tree,
        base=base,
        energy_per_bit=cfg.get("communication_energy_per_bit", 1e-4),
    )

    return ServerAppComponents(
        strategy=strategy,
        config=ServerConfig(num_rounds=cfg.get("num-server-rounds", 10)),
    )


# Instantiate Flower app
app = ServerApp(server_fn=server_fn)
