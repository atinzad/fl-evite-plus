# fl_evite_plus/client_app.py

import random
import warnings
from typing import Dict, Tuple, List

from sklearn.metrics import log_loss

from flwr.client import NumPyClient, ClientApp
from flwr.common import NDArrays, Scalar, Parameters

from fl_evite_plus.task import (
    load_data,
    get_model,
    get_model_params,
    set_model_params,
    set_initial_params,
)
from fl_evite_plus.server_app import load_tree_with_weights, all_nodes


class FlowerClient(NumPyClient):
    def __init__(self, model, Xtr, Xte, ytr, yte, cid: int):
        self.model = model
        self.X_train, self.X_test = Xtr, Xte
        self.y_train, self.y_test = ytr, yte
        self.cid = cid

    # ---------- FIT ------------------------------------------------------ #
    def fit(
        self,
        parameters: Parameters,
        config: Dict[str, Scalar],
    ) -> Tuple[List[NDArrays], int, Dict[str, Scalar]]:
        # 1. unpack & set
        set_model_params(self.model, parameters)

        # 2. train
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.model.fit(self.X_train, self.y_train)

        # 3. return exactly (params, num_examples, metrics)
        return (
            get_model_params(self.model),     # type: ignore[list]
            len(self.X_train),
            {"cid": self.cid},         # must include cid
        )

    # ---------- EVALUATE ------------------------------------------------- #
    def evaluate(
        self,
        parameters: Parameters,
        config: Dict[str, Scalar],
    ) -> Tuple[float, int, Dict[str, Scalar]]:
        # 1. unpack & set
        set_model_params(self.model, parameters)

        # 2. measure
        loss = log_loss(self.y_test, self.model.predict_proba(self.X_test))
        acc = self.model.score(self.X_test, self.y_test)

        # 3. return exactly (loss, num_examples, metrics)
        return (
            float(loss),
            len(self.X_test),
            {"cid": self.cid, "accuracy": float(acc)},
        )

class DummyClient(NumPyClient):
    def __init__(self, cid, dummy_params):
        self.cid = cid
        self.dummy_params = dummy_params

    def get_parameters(self, config): return self.dummy_params
    def fit(self, parameters, config): return self.dummy_params, 0, {"cid": self.cid}
    def evaluate(self, parameters, config): return 0.0, 0, {"cid": self.cid}

# ---------------- client_fn --------------------------------------------- #
def client_fn(context):
    cid = context.node_config["partition-id"]
    n_parts = context.node_config["num-partitions"]

    model = get_model(context.run_config["penalty"], context.run_config["local-epochs"])
    set_initial_params(model)
    dummy_params = get_model_params(model)

    # Optional: enforce client is part of the aggregation tree
    tree = load_tree_with_weights(context.run_config["mst_file"])
    valid_ids = all_nodes(tree) - {0}
    if cid not in valid_ids:
        return DummyClient(cid, dummy_params).to_client()

    Xtr, Xte, ytr, yte = load_data(cid, n_parts)
    mdl = get_model(context.run_config["penalty"], context.run_config["local-epochs"])
    set_initial_params(mdl)

    return FlowerClient(mdl, Xtr, Xte, ytr, yte, cid).to_client()


# Register with Flower
app = ClientApp(client_fn=client_fn)
