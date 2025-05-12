import re
import csv
import numpy as np
from pathlib import Path

logs_dir = Path("logs")
output_csv = Path("simulation_results.csv")

def extract_round_values(pattern: str, text: str):
    return [float(val) for _, val in re.findall(pattern, text)]

def extract_block(pattern: str, text: str):
    matches = re.findall(pattern, text, re.DOTALL)
    return matches[0] if matches else ""

parsed_results = []
max_rounds = 0

for log_file in sorted(logs_dir.glob("*.txt")):
    with open(log_file, "r") as f:
        content = f.read()

    file_name = log_file.stem

    if "Run finished" not in content:
        print(f"⚠️ Skipping incomplete or failed run: {file_name}")
        continue

    energy = extract_round_values(r"'round_total_energy':\s*\[\s*\((\d+), ([\d.]+)\)", content)
    avg_acc = extract_round_values(r"'round_avg_accuracy':\s*\[\s*\((\d+), ([\d.]+)\)", content)
    cent_acc = extract_round_values(r"'centralized_accuracy':\s*\[\s*\((\d+), ([\d.]+)\)", content)

    loss_block = extract_block(r"History \(loss, centralized\):(.+?)INFO", content)
    cent_loss = extract_round_values(r"round \d+: ([\d.]+)", loss_block)

    max_len = max(len(energy), len(avg_acc), len(cent_acc), len(cent_loss))
    max_rounds = max(max_rounds, max_len)

    row = [file_name]
    row += energy + [""] * (max_len - len(energy))
    row += avg_acc + [""] * (max_len - len(avg_acc))
    row += cent_acc + [""] * (max_len - len(cent_acc))
    row += cent_loss + [""] * (max_len - len(cent_loss))
    parsed_results.append(row)

# === HEADER ===
header = ["file_name"]
for r in range(1, max_rounds + 1):
    header.append(f"round_total_energy_{r}")
for r in range(1, max_rounds + 1):
    header.append(f"round_avg_accuracy_{r}")
for r in range(max_rounds + 1):
    header.append(f"centralized_accuracy_{r}")
for r in range(max_rounds + 1):
    header.append(f"loss_centralized_{r}")

# === WRITE CSV ===
with open(output_csv, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(header)
    writer.writerows(parsed_results)

print(f"\n📊 Results saved to: {output_csv.resolve()}")
