"""fl-evite-plus: A Flower / sklearn app."""


import warnings, random, csv, os
from pathlib import Path
from functools import lru_cache

from sklearn.metrics import log_loss

from flwr.client import ClientApp, NumPyClient
from flwr.common import Context
from fl_evite_plus.task import (
    get_model,
    get_model_params,
    load_data,
    set_initial_params,
    set_model_params,
)
from fl_evite_plus.comm_cost import energy_comm_cost  # Our energy cost function

#@lru_cache(maxsize=1)
def _load_distance_map(csv_path: str):
    print(f"csv_path {csv_path}")
    mapping = {}
    if csv_path and Path(csv_path).exists():
        with open(csv_path, newline="") as fh:
            for row in csv.DictReader(fh):
                mapping[int(row["client_id"])] = float(row["distance_m"])
    return mapping


class FlowerClient(NumPyClient):
    def __init__(self, model, X_train, X_test, y_train, y_test,cid):
        self.model = model
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.cid = cid

    def fit(self, parameters, config):
        set_model_params(self.model, parameters)

        # Simulate communication error with probability communication_error_rate
        error_rate = config.get("communication_error_rate", 0.0)
        if random.random() < error_rate:
            raise Exception("Simulated communication error in fit")


        new_params = get_model_params(self.model)
        
        # Retrieve energy and distance parameters from config with defaults as fallback.
        energy_per_bit = config.get("communication_energy_per_bit", 0.0001)

        print("config.get(distance_file)",config.get("distance_file", ""))
        distance_map = _load_distance_map(config.get("distance_file", ""))
        print(f"distance map is {distance_map}")
        print(f"config {config}")
        distance = distance_map.get(self.cid)
        print(f"distance {distance}")
        if distance is None:   # fallback if ID not present
            dmin = config.get("communication_distance_min", 5)
            dmax = config.get("communication_distance_max", 20)
            # Choose a random distance within the provided range.
            distance = random.uniform(dmin, dmax)
            print(f"distance format {distance}")
        
        
        # Calculate communication cost.
        comm_cost = energy_comm_cost(new_params, energy_per_bit, distance)

        # Ignore convergence warnings due to low local epochs
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.model.fit(self.X_train, self.y_train)

        # Return the updated parameters and metrics including the communication cost and chosen distance.
        return new_params, len(self.X_train), {"comm_cost": comm_cost, "distance": distance}

    def evaluate(self, parameters, config):
        set_model_params(self.model, parameters)

        # Optionally simulate communication error in evaluation as well.
        error_rate = config.get("communication_error_rate", 0.0)
        if random.random() < error_rate:
            raise Exception("Simulated communication error in evaluate")

        loss = log_loss(self.y_test, self.model.predict_proba(self.X_test))
        accuracy = self.model.score(self.X_test, self.y_test)

        return loss, len(self.X_test), {"accuracy": accuracy}


def client_fn(context: Context):
    cid= context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]

    X_train, X_test, y_train, y_test = load_data(cid, num_partitions)

    # Create LogisticRegression Model based on configuration parameters
    penalty = context.run_config["penalty"]
    local_epochs = context.run_config["local-epochs"]
    model = get_model(penalty, local_epochs)

    # Initialize model parameters
    set_initial_params(model)

    return FlowerClient(model, X_train, X_test, y_train, y_test, cid).to_client()


# Register Flower ClientApp
app = ClientApp(client_fn=client_fn)
