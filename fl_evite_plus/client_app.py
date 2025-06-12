# fl_evite_plus/client_app.py

import random
import warnings
from typing import Dict, Tuple, List

import numpy as np
from sklearn.metrics import log_loss

from flwr.client import NumPyClient, ClientApp
from flwr.common import NDArrays, Scalar, Parameters

from fl_evite_plus.task import (
    load_data,
    get_model,
    get_model_params,
    set_model_params,
    set_initial_params,
    get_client_labels,
)
from fl_evite_plus.server_app import load_tree_with_weights, all_nodes


class FlowerClient(NumPyClient):
    def __init__(
        self,
        model,
        Xtr,
        Xte,
        ytr,
        yte,
        cid: int,
        allowed_labels: List[int],
        k: int,
    ):
        self.model = model
        self.X_train, self.X_test = Xtr, Xte
        self.y_train, self.y_test = ytr, yte
        self.cid = cid
        self.allowed_labels = allowed_labels  # Exactly k labels
        self.k = k

    # ---------- FIT ------------------------------------------------------ #
    def fit(
        self,
        parameters: Parameters,
        config: Dict[str, Scalar],
    ) -> Tuple[List[NDArrays], int, Dict[str, Scalar]]:
        # 1. Unpack & set incoming global parameters
        set_model_params(self.model, parameters)

        # 2. Tell the model it should expect exactly k classes (the allowed_labels)
        self.model.classes_ = np.array(self.allowed_labels, dtype=int)

        # 3. Reinitialize coef_/intercept_ to match k × n_features,
        #    but derive n_features from X_train.shape[1] rather than from the old coef_
        n_features = self.X_train.shape[1]
        self.model.coef_ = np.zeros((self.k, n_features), dtype=self.model.coef_.dtype)
        if self.model.fit_intercept:
            self.model.intercept_ = np.zeros((self.k,), dtype=self.model.intercept_.dtype)

        # 4. Now train on the client’s filtered data (X_train contains only those k labels)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.model.fit(self.X_train, self.y_train)

        # 5. After training, “pad” those learned (k×n_features) rows back into a full (10×n_features)
        full_coef = np.zeros((10, n_features), dtype=self.model.coef_.dtype)
        if self.model.fit_intercept:
            full_intercept = np.zeros((10,), dtype=self.model.intercept_.dtype)
        else:
            full_intercept = None

        # Copy each of the k learned rows into the row index = label
        for i, lbl in enumerate(self.allowed_labels):
            full_coef[lbl, :] = self.model.coef_[i, :]
            if self.model.fit_intercept:
                full_intercept[lbl] = self.model.intercept_[i]

        # 6. Build the padded parameter list
        if self.model.fit_intercept:
            padded_params = [full_coef, full_intercept]
        else:
            padded_params = [full_coef]

        return (
            padded_params,
            len(self.X_train),
            {"cid": self.cid},
        )

    # ---------- EVALUATE ------------------------------------------------- #
    def evaluate(
        self,
        parameters: Parameters,
        config: Dict[str, Scalar],
    ) -> Tuple[float, int, Dict[str, Scalar]]:
        # 1. Unpack & set
        set_model_params(self.model, parameters)

        # 2. Measure—tell log_loss all 10 classes exist
        proba = self.model.predict_proba(self.X_test)
        loss = log_loss(self.y_test, proba, labels=np.arange(10))
        acc = self.model.score(self.X_test, self.y_test)

        return (
            float(loss),
            len(self.X_test),
            {"cid": self.cid, "accuracy": float(acc)},
        )


class DummyClient(NumPyClient):
    def __init__(self, cid, dummy_params):
        self.cid = cid
        self.dummy_params = dummy_params

    def get_parameters(self, config):
        return self.dummy_params

    def fit(self, parameters, config):
        return self.dummy_params, 0, {"cid": self.cid}

    def evaluate(self, parameters, config):
        return 0.0, 0, {"cid": self.cid}


# ---------------- client_fn --------------------------------------------- #
def client_fn(context):
    cid = context.node_config["partition-id"]
    n_parts = context.node_config["num-partitions"]

    # 1) Read k from run_config (labels-per-client)
    k = context.run_config.get("labels-per-client", 3)

    # 2) Prepare a fresh model to obtain “initial params” for DummyClient
    model = get_model(context.run_config["penalty"], context.run_config["local-epochs"])
    set_initial_params(model)
    dummy_params = get_model_params(model)

    # 3) (Optional) Filter by MST membership
    tree = load_tree_with_weights(context.run_config["mst_file"])
    valid_ids = all_nodes(tree) - {0}
    if cid not in valid_ids:
        return DummyClient(cid, dummy_params).to_client()

    # 4) Load the client’s data, passing in k so load_data filters exactly k labels
    Xtr, Xte, ytr, yte = load_data(cid, n_parts, k)
    allowed_labels = get_client_labels(cid, k)
    #allowed_labels = get_client_labels(cid, k, round_number=server_round)

    # 5) If this client has no training examples, return DummyClient
    if Xtr.shape[0] == 0 or ytr.shape[0] == 0:
        return DummyClient(cid, dummy_params).to_client()

    # 6) Otherwise, create a FlowerClient that knows its k labels
    mdl = get_model(context.run_config["penalty"], context.run_config["local-epochs"])
    set_initial_params(mdl)  # will be overwritten in fit()
    return FlowerClient(mdl, Xtr, Xte, ytr, yte, cid, allowed_labels, k).to_client()


# Register with Flower
app = ClientApp(client_fn=client_fn)
