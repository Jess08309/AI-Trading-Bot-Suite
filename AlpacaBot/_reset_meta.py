"""One-time script to reset corrupted meta_learner state."""
import json

path = r"C:\AlpacaBot\data\state\meta_learner.json"
with open(path, "r") as f:
    data = json.load(f)

# Reset corrupted thresholds to defaults
data["confidence_threshold"] = 0.68
data["min_rule_score"] = 6
# Reset weights to defaults (current weights skewed by garbage 0.5 data)
data["source_weights"] = {"ml_model": 0.35, "sentiment": 0.15, "rule_score": 0.50}
# Clear corrupted history (all ml_model entries are 0.5 dummy data from scanner bug)
data["source_history"] = {"ml_model": [], "sentiment": [], "rule_score": []}
data["threshold_history"] = []

with open(path, "w") as f:
    json.dump(data, f, indent=2)

print("Meta-learner state RESET:")
print(f"  confidence_threshold: 0.68 (was 0.76)")
print(f"  min_rule_score: 6 (was 8)")
print(f"  weights: ml=0.35, sent=0.15, rule=0.50")
print(f"  history: cleared (was all 0.5 garbage)")
