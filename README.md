# Oldenborg Autonomous Navigation

Code for training machine learning models to navigate our Oldenborg simulation environment.

- `boxnav`: generate datasets using a map and Unreal Engin
- `jetbot`: run inference on a JetBot
- `learning`: train models to navigate the simulation
- `ue5osc`: communicate between Unreal Engine (or packaged simulations) and Python

Setup

~~~bash
# Create the environment
mamba env create -f arcs-su25.yml
mamba activate arcs-su25

# Install boxnav (when collecting data or running simulation-based inference)
cd boxnav
python -m pip install --editable .

# Install ue5osc (when collecting data for running simulation-based inference)
cd ue5osc
python -m pip install --editable .
~~~

## Workflow

1. Generate data using `boxnav/boxsim.py`
2. Upload data using `learning/upload_data.py`.
3. Train model using `learning/training.py`.
4. Perform inference using `learning/inference.py`.

For example,

~~~bash
# Activate the environment
conda activate ENVIRONMENT

# Runs a navigator in Python and the simulation environment
# This will run on a system that can run Unreal Engine
# Open unreal engine before running code (ie. ARCSAssets.exe)
#   This will steal your mouse, run on split screen and use Alt+tab to navigate between screens
cd boxnav
python boxsim.py NAVIGATOR --image_directory IMAGE_DIRECTORY

# Uploads the dataset to the server
# You can upload from wherever the data is generated (probably the same system as above)
# For first time: Terminal will ask you to log-in to wandb
#   When asked to authorize, authentication code will be hidden when pasted so hit ctrl+v once and hit enter
cd learning
python upload_data.py DATA_NAME PROJECT_NAME "Sample description of uploading run..." IMAGE_DIRECTORY

# Trains the model
# This should be run on a system with a GPU (e.g., our server)
# training.py, sbatch scripts, and datasets should be in the same directory on the server (could be learning)
cd learning
python training.py MODEL_NAME PROJECT_NAME "Sample description of training run..." ARCHITECTURE_NAME DATA_NAME(S) --local_data

# Performs inference
# This will run on a system that can run Unreal Engine
# Note: MODEL_NAME_FROM_WANDB can be found in arcslaboratory -> Projects -> PROJECT_NAME -> Artifacts
cd learning
python inference.py INFERENCE_NAME PROJECT_NAME "Sample description of inference run..." MODEL_NAME_FROM_WANDB:VERSION IMAGE_SAVE_FOLDER_NAME
~~~

Notice that the first program argument for uploading, training, and inference is the name of the artifact created by that step.

## Notes

Some things to do

- Work through the TODOs
- Consolidate all training shell scripts into one that takes arguments
- Clean up calibration script
- Remove jetbot utils script?
- Add information on using sbatch scripts
- Create a separate environment file for the JetBot
