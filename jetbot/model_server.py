from dataclasses import dataclass
from pathlib import Path

import tyro
import wandb
from fastai.callback.wandb import WandbCallback
from fastai.learner import load_learner
from rpc import RPCServer
from utils import y_from_filename  # noqa: F401 (needed for fastai load_learner)


@dataclass
class Args:
    wandb_name: str
    wandb_project: str
    wandb_notes: str
    wandb_model: str


def model_run(image_filename):
    "Compute the action to take given an image filename."
    action_to_take, action_index, action_probs = model.predict(image_filename)
    action_prob = action_probs[action_index]
    print(action_prob)
    print(" " + action_to_take + " \n")
    return action_to_take


args = tyro.cli(Args)


# TODO: allow user to specify the RPC server address and port
server = RPCServer()

run = wandb.init(
    job_type="inference",
    entity="arcslaboratory",
    name=args.wandb_name,
    project=args.wandb_project,
    notes=args.wandb_notes,
)

# Ensure the wandb run was successfully initialized
if run is None:
    raise Exception("wandb.init() failed")

# Download the latest model artifact from Wandb
artifact = run.use_artifact(f"{args.wandb_model}:latest", type="model")
model_dir = artifact.download()
model_filename = Path(model_dir) / args.wandb_model
model_filename = str(model_filename) + ".pkl"
model_filename = Path(model_filename)

# Load the trained model using fastai's load_learner (and remove WandbCallback for inference)
model = load_learner(model_filename)
model.remove_cb(WandbCallback)

# Register the model_run function and start the RPC server
server.registerMethod(model_run)
server.run()
