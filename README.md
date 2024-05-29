# Oldenborg Autonomous Navigation

Code for training machine learning models to navigate our Oldenborg simulation environment.

- `ue5osc`: Code to communicate between Unreal Engine 5 (or packaged simulations) and Python
- `boxnav`: Code to compute "correct" actions based on the agent's location in simulation
- `learning`: Code to train machine learning models to navigate the simulation


## Workflow

1. Generate data using `boxnav/boxsim.py`
2. Upload data using `learning/upload_data.py`.
3. Train model using `learning/training.py`.
4. Perform inference using `learning/inference.py`.

For example,

~~~bash
# Runs a navigator in Python and the simulation environment
# This will run on a system that can run Unreal Engine
cd boxnav
python boxsim.py wandering --save_images data/

# Uploads the dataset to the server
# You can upload from wherever the data is generated (probably the same system as above)
cd learning
python upload_data.py PerfectStaticData TestingWorkflow "I am using this project to test the upload, train, then inference workflow." data/

# Trains the model
# This should be run on a system with a GPU (e.g., our server)
cd learning
python training.py PerfectStaticModel TestingWorkflow "Testing training..." resnet18 PerfectStaticData

# Performs inference
# This will run on a system that can run Unreal Engine
cd learning
python inference.py PerfectStaticInference TestingWorkflow "Testing inference..." PerfectStaticModel-resnet18-PerfectStaticData-rep00 InferenceImages
~~~

Notice that the first program argument for uploading, training, and inference is the name of the artifact created by that step.
