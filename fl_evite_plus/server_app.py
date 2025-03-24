"""fl-evite-plus: A Flower / sklearn app."""

from flwr.common import Context, ndarrays_to_parameters
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg
from fl_evite_plus.task import get_model, get_model_params, set_initial_params

def fit_metrics_aggregation_fn(results):
    """
    Aggregates fit metrics from clients.
    Expects results as a list of tuples (num_examples, metrics).
    Sums up the 'comm_cost' reported by each client.
    """
    total_energy = sum(
        metrics.get("comm_cost", 0.0)
        for _, metrics in results if metrics is not None
    )
    # Return a dict; this aggregated metric is now logged per round.
    return {"round_total_energy": total_energy}

def server_fn(context: Context):
    run_config = context.run_config
    num_rounds = run_config.get("num-server-rounds", 3)
    fraction_fit = run_config.get("fraction_fit", 1.0)
    fraction_evaluate = run_config.get("fraction_evaluate", 1.0)

    # Create the LogisticRegression Model using configuration parameters.
    penalty = run_config.get("penalty", "l2")
    local_epochs = run_config.get("local-epochs", 1)
    model = get_model(penalty, local_epochs)

    # Initialize model parameters.
    set_initial_params(model)

    initial_parameters = ndarrays_to_parameters(get_model_params(model))

    # Define the FedAvg strategy with custom fractions and aggregation.
    strategy = FedAvg(
        fraction_fit=fraction_fit,
        fraction_evaluate=fraction_evaluate,
        min_available_clients=2,
        initial_parameters=initial_parameters,
        fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,  # Fixes the warning.
    )
    config = ServerConfig(num_rounds=num_rounds)

    return ServerAppComponents(strategy=strategy, config=config)

# Create and register the ServerApp
app = ServerApp(server_fn=server_fn)
