"""fl-evite-plus: A Flower / sklearn app."""

from flwr.common import Context, ndarrays_to_parameters
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg

from fl_evite_plus.task import (
    get_model,
    get_model_params,
    set_initial_params,
)



def fit_metrics_aggregation_fn(results):
    """
    Aggregates fit metrics returned by clients.

    Args:
        results (List[Tuple[int, Dict[str, Scalar]]]): 
            A list of (num_examples, metrics) for successful client updates.

    Returns:
        Dict with aggregated metrics for this training round.
    """
    total_energy = sum(
        metrics.get("comm_cost", 0.0)
        for _, metrics in results
        if metrics is not None
    )
    return {"round_total_energy": total_energy}


def evaluate_metrics_aggregation_fn(results):
    """
    Aggregates evaluate metrics returned by clients.

    Args:
        results (List[Tuple[int, Dict[str, Scalar]]]): 
            A list of (num_examples, metrics) for successful client evaluations.

    Returns:
        Dict with aggregated metrics for this evaluation round.
    """
    # Example: average accuracy
    accuracies = [
        metrics.get("accuracy", 0.0)
        for _, metrics in results
        if metrics is not None
    ]
    avg_accuracy = sum(accuracies) / len(accuracies) if accuracies else 0.0

    return {"round_avg_accuracy": avg_accuracy}


def server_fn(context: Context):
    """
    Server function that initializes and returns the ServerAppComponents.
    Reads config from context and sets up a FedAvg strategy with custom aggregation.
    """
    run_config = context.run_config

    def fit_config(server_round: int):
        """Dict sent to every client before it calls fit()."""

    
        return {
        # pass all keys the client expects
        "distance_file": run_config.get("distance_file", ""),
        "communication_error_rate": run_config.get("communication_error_rate", 0.0),
        "communication_energy_per_bit": run_config.get("communication_energy_per_bit", 1e-4),
        "communication_distance_min": run_config.get("communication_distance_min", 5),
        "communication_distance_max": run_config.get("communication_distance_max", 20),
        }
    
    def eval_config(server_round: int):
        """Dict sent before evaluate().  Often identical to fit_config."""
        return {
        "distance_file": run_config.get("distance_file", ""),
        "communication_error_rate": run_config.get("communication_error_rate", 0.0),
        }


    # Read core config
    num_rounds = run_config.get("num-server-rounds", 3)
    fraction_fit = run_config.get("fraction_fit", 1.0)
    fraction_evaluate = run_config.get("fraction_evaluate", 1.0)

    # Create and initialize the LogisticRegression model
    penalty = run_config.get("penalty", "l2")
    local_epochs = run_config.get("local-epochs", 1)
    model = get_model(penalty, local_epochs)
    set_initial_params(model)

    # Extract initial parameters for the strategy
    initial_parameters = ndarrays_to_parameters(get_model_params(model))

    # Define FedAvg strategy with aggregator functions
    strategy = FedAvg(
        fraction_fit=fraction_fit,
        fraction_evaluate=fraction_evaluate,
        min_available_clients=2,
        initial_parameters=initial_parameters,
        fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,
        evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn,
        on_fit_config_fn=fit_config,           # <── new
        on_evaluate_config_fn=eval_config,     # <── new
        )

    # Create server configuration
    config = ServerConfig(num_rounds=num_rounds)

    return ServerAppComponents(strategy=strategy, config=config)


# Create the Flower server app
app = ServerApp(server_fn=server_fn)
