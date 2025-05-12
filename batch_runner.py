import os
import subprocess
from pathlib import Path
import re
import csv
import numpy as np

# === CONFIGURATION ===
trees_dir = Path("generated_trees")
logs_dir = Path("logs")
output_csv = Path("simulation_results.csv")
logs_dir.mkdir(exist_ok=True)

# === UTILS ===
def extract_round_values(pattern: str, text: str):
    return [float(val) for _, val in re.findall(pattern, text)]

def extract_block(pattern: str, text: str):
    matches = re.findall(pattern, text, re.DOTALL)
    return matches[0] if matches else ""

def run_simulation(mst_path: Path, log_path: Path):
    env = os.environ.copy()
    env["MST_FILE"] = str(mst_path.resolve())

     # Kill any leftover Ray processes
    subprocess.run("ray stop", shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


    print(f"🚀 Running simulation for {mst_path.name}")
    with open(log_path, "w") as log_file:
        result = subprocess.run("flwr run", shell=True, env=env, stdout=log_file, stderr=subprocess.STDOUT)
    return result.returncode == 0

# === STEP 1: SERIAL RUNS ===
mst_files = sorted(trees_dir.glob("*.json"))
for mst_file in mst_files:
    log_file = logs_dir / f"{mst_file.stem}.txt"

    # Skip if log already exists and is successful
    if log_file.exists():
        with open(log_file) as f:
            if "Run finished" in f.read():
                print(f"🟡 Skipping (already done): {mst_file.name}")
                continue

    success = run_simulation(mst_file, log_file)
    if not success:
        print(f"❌ Simulation failed for: {mst_file.name}")

# === STEP 2: PARSE LOGS ===
parsed_results = []
max_rounds = 0

for log_file in sorted(logs_dir.glob("*.txt")):
    with open(log_file, "r") as f:
        content = f.read()

    

    file_name = log_file.stem

    energy = extract_round_values(r"'round_total_energy':\s*\[\s*\((\d+), ([\d.]+)\)", content)
    avg_acc = extract_round_values(r"'round_avg_accuracy':\s*\[\s*\((\d+), ([\d.]+)\)", content)
    cent_acc = extract_round_values(r"'centralized_accuracy':\s*\[\s*\((\d+), ([\d.]+)\)", content)

    loss_block = extract_block(r"History \(loss, centralized\):(.+?)INFO", content)
    cent_loss = extract_round_values(r"round \d+: ([\d.]+)", loss_block)

    max_len = max(len(energy), len(avg_acc), len(cent_acc), len(cent_loss))
    max_rounds = max(max_rounds, max_len)

    if all(np.isnan(v) or v == 0 for v in cent_acc) and all(np.isnan(v) or v == 0 for v in cent_loss):
        print(f"⚠️ Skipping {file_name}: No valid results (likely NaNs)")
        continue

    row = [file_name]
    row += energy + [""] * (max_len - len(energy))
    row += avg_acc + [""] * (max_len - len(avg_acc))
    row += cent_acc + [""] * (max_len - len(cent_acc))
    row += cent_loss + [""] * (max_len - len(cent_loss))

    parsed_results.append(row)

# === STEP 3: WRITE CSV ===
header = ["file_name"]
for r in range(1, max_rounds + 1):
    header.append(f"round_total_energy_{r}")
for r in range(1, max_rounds + 1):
    header.append(f"round_avg_accuracy_{r}")
for r in range(max_rounds + 1):
    header.append(f"centralized_accuracy_{r}")
for r in range(max_rounds + 1):
    header.append(f"loss_centralized_{r}")

with open(output_csv, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(header)
    writer.writerows(parsed_results)

print(f"\n📊 Results saved to: {output_csv.resolve()}")
