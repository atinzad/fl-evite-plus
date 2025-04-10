# fl-evite-plus: A Flower-based Federated Learning Simulation

`fl-evite-plus` is a federated learning application built using the [Flower](https://flower.ai/) framework and `scikit-learn`. It simulates federated training across multiple IoT-like client devices, incorporating customizable energy and communication models.

---

## 📁 Project Structure

```
fl-evite-plus/
├── README.md                   # Project documentation and instructions
├── pyproject.toml              # Project dependencies and Flower configuration
└── fl_evite_plus/              # Core application source code
    ├── __init__.py             # Package initialization
    ├── client_app.py           # Flower client logic (IoT device simulation)
    ├── server_app.py           # Flower server logic (aggregator)
    ├── task.py                 # Utilities: dataset loading, model handling
    └── comm_cost.py            # Communication energy cost calculation
```

---

## 🚀 Setting Up the Project (First Time)

1. **Clone the Repository**

```bash
git clone <repository-url>
cd fl-evite-plus
```

2. **Install Dependencies (Using `uv` from Astral)**

Ensure [`uv`](https://github.com/astral-sh/uv) is installed:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Install project dependencies:

```bash
uv venv
source .venv/bin/activate
uv pip install -e .
```

---

## 🖥️ Running the Simulation

Run the federated learning simulation locally:

```bash
flwr run .
```

### How to Read Output

The command output will include:

- **Round number and state**:
  - `[ROUND X]`: Indicates training round number.
  - `aggregate_fit`: Reports the number of clients successfully aggregated.
  - `aggregate_evaluate`: Evaluation results aggregation.

- **Metrics**:
  - **Loss**: Training loss per round.
  - **Communication Energy**: Energy consumed per round (Joules), logged as `round_total_energy`.

Example output:
```
[ROUND 1]
aggregate_fit: received 80 results and 0 failures
aggregate_evaluate: received 70 results and 0 failures
fit_metrics: {'round_total_energy': 362149.63}
```

---

## ⚙️ Configuring Simulation Parameters (`pyproject.toml`)

Adjust simulation parameters in `pyproject.toml` under `[tool.flwr.app.config]`:

| Parameter                      | Type    | Unit                  | Description |
|--------------------------------|---------|-----------------------|-------------|
| `num-server-rounds`            | Integer | Count (unitless)      | Number of training rounds to perform |
| `penalty`                      | String  | Unitless              | Regularization for logistic regression (`"l1"` or `"l2"`) |
| `local-epochs`                 | Integer | Count (unitless)      | Epochs each client trains locally per round |
| `communication_energy_per_bit` | Float   | Joules per bit (J/bit)| Energy consumption per transmitted bit |
| `communication_distance_min`   | Float   | Meters (m)            | Min. distance client-to-server |
| `communication_distance_max`   | Float   | Meters (m)            | Max. distance client-to-server |
| `communication_error_rate`     | Float   | Probability (unitless)| Chance a client's communication fails |
| `fraction_fit`                 | Float   | Probability (unitless)| Fraction of clients selected for training per round |
| `fraction_evaluate`            | Float   | Probability (unitless)| Fraction of clients selected for evaluation per round |

---

## 🛠️ Example Configuration

```toml
[tool.flwr.app.config]
num-server-rounds = 3
penalty = "l2"
local-epochs = 1
communication_energy_per_bit = 0.0001
communication_distance_min = 5
communication_distance_max = 20
communication_error_rate = 0.1
fraction_fit = 0.8
fraction_evaluate = 0.7
```

---

## 📝 Additional Resources

- [Flower Documentation](https://flower.ai/docs)
- [Astral UV Tool](https://github.com/astral-sh/uv)

---

Happy Federating! 🌼📡✨

