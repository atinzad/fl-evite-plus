# fl_evite_plus/server_app.py
"""Hierarchical FL server (distances embedded in mst.json)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set

import numpy as np
import ray

from fl_evite_plus.task import get_model, set_model_params, load_data, load_full_test_data
from sklearn.metrics import accuracy_score, log_loss


from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.client_manager import ClientProxy
from flwr.server.strategy import FedAvg, Strategy
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

from fl_evite_plus.task import get_model, get_model_params, set_initial_params

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
    """Bottom‐up aggregation with server‐side energy accounting."""

    def __init__(
        self,
        weight_tree: Dict[int, Dict[int, float]],
        base: FedAvg,
        energy_per_bit: float,
    ):
        super().__init__()
        self.tree = weight_tree
        self.base = base
        self.e_bit = energy_per_bit
        self.valid_cids = all_nodes(weight_tree) - {0}

    def initialize_parameters(self, client_manager) -> Optional[Parameters]:
        return self.base.initialize_parameters(client_manager)

    def configure_fit(
        self,
        server_round: int,
        parameters: Parameters,
        client_manager,
    ) -> List[Tuple[ClientProxy, FitIns]]:
        clients = list(client_manager.all().values())  # Dict[str, ClientProxy] → [ClientProxy]
        fit_ins = FitIns(parameters, {})
        return [(c, fit_ins) for c in clients]


    def configure_evaluate(
        self,
        server_round: int,
        parameters: Parameters,
        client_manager,
    ) -> List[Tuple[ClientProxy, EvaluateIns]]:
        clients = list(client_manager.all().values())
        eval_ins = EvaluateIns(parameters, {})
        return [(c, eval_ins) for c in clients]



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
            mapping[int(cid)] = (nds, n_examples)
        return mapping

    def _aggregate_subtree(
        self,
        node: int,
        parent: Optional[int],
        updates: Dict[int, Tuple[List[np.ndarray], int]],
    ) -> Tuple[List[np.ndarray], int, float]:
        triplets: List[Tuple[List[np.ndarray], int, float]] = []

        for child in self.tree.get(node, {}):
            child_params, child_n, child_e = self._aggregate_subtree(child, node, updates)
            triplets.append((child_params, child_n, child_e))

        if node in updates:
            params, n = updates[node]
            energy = 0.0
            if parent is not None:
                dist = self.tree[parent][node]
                energy = self.e_bit * (dist**2) * param_bits(params)
            triplets.append((params, n, energy))

        if not triplets:
            raise ValueError(f"No update for node {node}")

        weight_tuples = [ (p[0], n) for (p, n, _) in triplets ]
        agg_weights = aggregate(weight_tuples)
        total_n = sum(n for _, n in weight_tuples)


        if len(triplets[0][0]) > 1:
            intercept_tuples = [ (p[1], n) for (p, n, _) in triplets ]
            agg_intercept = aggregate(intercept_tuples)
            agg_params = [agg_weights, agg_intercept]
        else:
            agg_params = [agg_weights]

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
        accs = [ ev_res.metrics.get("accuracy", 0.0) for _, ev_res in results if ev_res.metrics ]
        avg_acc = float(sum(accs) / len(accs)) if accs else 0.0
        return 0.0, {"round_avg_accuracy": avg_acc}
    
 

    def evaluate(
        self,
        server_round: int,
        parameters: Parameters,
        ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        # Use a fixed partition for centralized eval, or merge partitions if available
        #_, X_test, _, y_test = load_data(partition_id=0, num_partitions=1)
        X_test, y_test = load_full_test_data(num_parts=10)

        # Initialize model
        model = get_model(penalty="l2", local_epochs=1)  # Match training config
        set_initial_params(model) 
        ndarrays = parameters_to_ndarrays(parameters)
        set_model_params(model, ndarrays)

        # Evaluate
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)

        loss = log_loss(y_test, y_proba)
        acc = accuracy_score(y_test, y_pred)

        return float(loss), {"centralized_accuracy": float(acc)}



# --------------------------------------------------------------------------- #
#  ServerApp entry point                                                      #
# --------------------------------------------------------------------------- #
def server_fn(context: Context) -> ServerAppComponents:
    cfg = context.run_config
    tree = load_tree_with_weights(cfg["mst_file"])
    num_clients = len(all_nodes(tree))

    model = get_model(cfg["penalty"], cfg["local-epochs"])
    set_initial_params(model)
    initial_parameters = ndarrays_to_parameters(get_model_params(model))

    def fit_cfg(_rnd: int) -> Dict[str, object]:
        return {
            "communication_error_rate": cfg.get("communication_error_rate", 0.0),
            "mst_file": cfg["mst_file"],
        }

    def eval_cfg(_rnd: int) -> Dict[str, object]:
        return {
            "communication_error_rate": cfg.get("communication_error_rate", 0.0),
            "mst_file": cfg["mst_file"],
        }

    base = FedAvg(
        fraction_fit=cfg.get("fraction_fit", 1.0),
        fraction_evaluate=cfg.get("fraction_evaluate", 1.0),
        min_available_clients=num_clients,
        initial_parameters=initial_parameters,
        on_fit_config_fn=fit_cfg,
        on_evaluate_config_fn=eval_cfg,
    )

    strategy = HierarchicalStrategy(
        weight_tree=tree,
        base=base,
        energy_per_bit=cfg.get("communication_energy_per_bit", 1e-4),
    )

    return ServerAppComponents(
        strategy=strategy,
        config=ServerConfig(num_rounds=cfg.get("num-server-rounds", 3)),
    )

# Instantiate Flower app
app = ServerApp(server_fn=server_fn)
