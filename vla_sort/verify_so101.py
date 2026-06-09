"""Verify the SO-101 config registered & builds (run with `uv run python` in openpi env)."""
from openpi.training import config as c

names = [x.name for x in c._CONFIGS]
assert "pi05_so101_sort_lora" in names, f"NOT registered! configs with so101: {[n for n in names if 'so101' in n]}"
print("REGISTERED OK. so101 configs:", [n for n in names if "so101" in n])

# Build the data config to surface transform/key errors before training.
cfg = next(x for x in c._CONFIGS if x.name == "pi05_so101_sort_lora")
print("model:", type(cfg.model).__name__, "pi05=", getattr(cfg.model, "pi05", None))
print("data factory:", type(cfg.data).__name__, "repo_id=", cfg.data.repo_id)
print("batch_size:", cfg.batch_size, "num_train_steps:", cfg.num_train_steps)
print("OK")
