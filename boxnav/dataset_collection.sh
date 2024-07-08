#!/usr/bin/env bash

# start in boxnav directory
# have Git Bash installed
# open new terminal ('Git Bash' from '+' dropdown menu)
# make script executabe if you haven't already ('chmod +x dataset_collection.sh' in terminal)
# run script ('./dataset_collection.sh' in terminal)

# python boxsim.py TELEPORTING --max_total_actions 100000 --image_directory Teleporting100kRandEvery10Data --randomize_interval 10
# python boxsim.py TELEPORTING --max_total_actions 100000 --image_directory Teleporting100kRandEvery50Data --randomize_interval 50

cd ..
cd learning

python upload_data.py Teleporting100kRandEvery10Data Summer2024Official "Uploading 100,000 images of the Oldenborg environment captured using the Teleporting navigator with randomization interval 10." ../boxnav/Teleporting100kRandEvery10Data
python upload_data.py Teleporting100kRandEvery50Data Summer2024Official "Uploading 100,000 images of the Oldenborg environment captured using the Teleporting navigator with randomization interval 50." ../boxnav/Teleporting100kRandEvery50Data