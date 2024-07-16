#!/usr/bin/env bash

# start in boxnav directory
# have Git Bash installed
# open new terminal ('Git Bash' from '+' dropdown menu)
# make script executabe if you haven't already ('chmod +x inference_wandering.sh' in terminal)
# run script ('./inference_wandering.sh' in terminal)

python inference.py PerfectTestInference Summer2024Official "Test how many actions it takes for Perfect model to get through environment" PerfectStaticModel-ResNet18-Perfect100kData-rep00 PerfectInference --max_actions 1000 --num_trials 10