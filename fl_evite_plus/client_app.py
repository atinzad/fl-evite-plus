"""fl-evite-plus: A Flower / sklearn app."""

import warnings
import random

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
from fl_evite_plus.comm_cost import energy_comm_cost  # Import the cost function

class FlowerClient(NumPyClient):
    def __init__(self, model, X_train, X_test, y_train, y_test):
        self.model = model
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

    def fit(self, parameters, config):
        set_model_params(self.model, parameters)

        # Ignore convergence failure due to low local epochs
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.model.fit(self.X_train, self.y_train)

        new_params = get_model_params(self.model)
        
        # Retrieve communication parameters from config with defaults as fallback.
        energy_per_bit = config.get("communication_energy_per_bit", 0.0001)
        distance_min = config.get("communication_distance_min", 5)
        distance_max = config.get("communication_distance_max", 20)
        
        # Choose a random distance within the provided range.
        distance = random.uniform(distance_min, distance_max)
        
        # Calculate communication cost using the energy * (distance^2) model.
        comm_cost = energy_comm_cost(new_params, energy_per_bit, distance)

        # Return the new parameters and a metrics dict including cost and chosen distance.
        return new_params, len(self.X_train), {"comm_cost": comm_cost, "distance": distance}

    def evaluate(self, parameters, config):
        set_model_params(self.model, parameters)

        loss = log_loss(self.y_test, self.model.predict_proba(self.X_test))
        accuracy = self.model.score(self.X_test, self.y_test)

        return loss, len(self.X_test), {"accuracy": accuracy}


def client_fn(context: Context):
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]

    X_train, X_test, y_train, y_test = load_data(partition_id, num_partitions)

    # Create LogisticRegression Model
    penalty = context.run_config["penalty"]
    local_epochs = context.run_config["local-epochs"]
    model = get_model(penalty, local_epochs)

    # Setting initial parameters, akin to model.compile for keras models
    set_initial_params(model)

    return FlowerClient(model, X_train, X_test, y_train, y_test).to_client()


# Flower ClientApp
app = ClientApp(client_fn=client_fn)
