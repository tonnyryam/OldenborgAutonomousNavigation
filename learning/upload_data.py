from argparse import ArgumentParser

import wandb

wandb.login()
api = wandb.Api()
projects = api.projects(entity="arcslaboratory")
project_names = [project.name for project in projects]

arg_parser = ArgumentParser("Upload the given dataset to wandb.")

arg_parser.add_argument("wandb_name", help="Name of run and artifact.")
arg_parser.add_argument("wandb_project", help="Wandb project name.")
arg_parser.add_argument("wandb_notes", help="Wandb run description.")
arg_parser.add_argument("data_dir", help="Directory containing dataset.")

args = arg_parser.parse_args()

while args.wandb_project not in project_names:
    name = input(
        "\n\nThe project name you entered is not in wandb entity 'arcslaboratory'.\n\nEnter the name of an existing project to upload to ("
        + ", ".join(project_names)
        + ")...\n\nOR\n\n...press ENTER to create a new project using the previouly entered name ('"
        + args.wandb_project
        + "').\n\n"
    )
    if name in project_names:
        args.wandb_project = name
    if name == "":
        project_names.append(args.wandb_project)


run = wandb.init(
    job_type="dataset-upload",
    entity="arcslaboratory",
    project=args.wandb_project,
    notes=args.wandb_notes,
)

if run is None:
    raise Exception("wandb.init() failed")

artifact = wandb.Artifact(name=args.wandb_name, type="dataset")
artifact.add_dir(local_path=args.data_dir, name="data")
run.log_artifact(artifact)
