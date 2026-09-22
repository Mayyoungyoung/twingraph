"""Train explicit procedural imitation policies, with honest held-out metrics."""
from pathlib import Path
import json
from simbench.assembly.sensor_learning_v12 import train
from simbench.assembly.wiping import train_v12

if __name__ == "__main__":
    directory = Path("simbench/assembly/checkpoints")
    insertion = train(directory / "insert_sensor_bc_v12.npz")
    wiping = train_v12(directory / "wipe_imitation_v12.npz")
    print(json.dumps({"insertion_action_mse": insertion["heldout_action_mse"],
                      "wipe_task_rmse_m": [x["path_rmse_m"] for x in wiping["heldout_tasks"]],
                      "physical_success_claim": False}, indent=2))
