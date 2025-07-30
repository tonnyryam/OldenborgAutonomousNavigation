from dataclasses import dataclass
from pathlib import Path

import tyro
import wandb
from rpc import RPCServer

from timm import create_model
import torch
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from PIL import Image


@dataclass
class Args:
    wandb_name: str
    wandb_project: str
    wandb_notes: str
    local_model: str
    arch: str


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

model_path = args.local_model
arch = args.arch
model = create_model(arch, pretrained=False, num_classes=3)

checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
if "state_dict" in checkpoint:
    model.load_state_dict(checkpoint["state_dict"])
else:
    model.load_state_dict(checkpoint)
    
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() and torch.backends.cuda.is_built() else "cpu")
model.to(device)


config = resolve_data_config({}, model=model)
transform = create_transform(**config)

def model_run(image_filename):
    "Compute the action to take given an image filename."
    img = Image.open(image_filename).convert("RGB")
    tensor = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(tensor)
        prob = torch.nn.functional.softmax(output, dim=1)
        pred = torch.argmax(prob, dim=1).item()
    return pred


# Register the model_run function and start the RPC server
server.registerMethod(model_run)
server.run()
