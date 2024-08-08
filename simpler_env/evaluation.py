import json
import argparse
import os
import base64

import mediapy as media
import numpy as np
import tensorflow as tf
import imageio
from PIL import Image

import simpler_env
from simpler_env import ENVIRONMENTS

from reasoning_utils import generate_reasoning_image, resize_image, extract_number
from simpler_env.policies.openvla.vla_model import OpenVLAInference
import time

from pprint import pprint

# images_path = "/iliad/group/datasets/bridgedata_v2/raw/bridge_data_v2/datacol1_toykitchen1/many_skills/0/2023-03-15_14-35-28/raw/traj_group0/traj" +str(tj) + "/images0"
# file_path = "/iliad/group/datasets/bridgedata_v2/raw/bridge_data_v2/datacol1_toykitchen1/many_skills/0/2023-03-15_14-35-28/raw/traj_group0/traj" +str(tj) + "/lang.txt"

TRAJ_DIR = "/iliad/group/datasets/bridgedata_v2/raw/bridge_data_v2/datacol1_toykitchen1/many_skills/0/2023-03-15_14-35-28/raw/traj_group0"


def main():
    sorted_folders = sorted(os.listdir(TRAJ_DIR))
    results = {}
    print(sorted_folders)
    # Load existing data if the file exists
    results_file = 'forced_cot_results.json'
    try:
        with open(results_file, 'r') as f:
            results = json.load(f)
    except (json.JSONDecodeError, FileNotFoundError):
        # Handle the error or initialize an empty dictionary
        print("Error reading JSON file. Initializing an empty results dictionary.")
        results = {}

    for folder in sorted_folders:
        if folder.startswith("traj")\
            and (folder not in results or len(list(results[folder].keys())) < 2): # Process only new folders
            traj_dir = os.path.join(TRAJ_DIR, folder)
            images_path = os.path.join(traj_dir, "images0")
            file_path = os.path.join(traj_dir, "lang.txt")
            
            print(f"Processing {traj_dir}")
            # Check if lang.txt file exists
            if not os.path.exists(file_path):
                print(f"lang.txt not found in {traj_dir}, skipping this trajectory.")
                continue
            
            image_list = []
            for filename in sorted(os.listdir(images_path), key=extract_number):
                if filename.endswith(".jpg"):
                    img_path = os.path.join(images_path, filename)
                    img = Image.open(img_path).convert('RGB')  # Ensure image is in RGB format
                    img_array = np.array(img, dtype=np.uint8)
                    image_list.append(img_array)

            for model_name in ["vla", "ECoT"]:
                start_time = time.time()
                
                model_text_name = "ECoT" if model_name == "ECoT" else "OpenVLA"
                model = OpenVLAInference(model=model_name, policy_setup="widowx_bridge")
                
                if folder not in results:
                    results[folder] = {}
                    
                if len(list(results[folder].keys())) == 2:
                    print("Already finished collecting the data for this setup")
                    continue
                
                with open(file_path, 'r') as file:
                    instruction = file.readline().strip()
                
                model.reset(instruction)
                
                model_results = {
                    "instruction": instruction,
                    "actions": [],
                    "time_taken": 0
                }
                
                predicted_terminated, success, truncated = False, False, False
                
                for timestep in range(len(image_list)):
                    image = image_list[timestep]
                    image = resize_image(image, (256, 256))
                    raw_action, action, generated_text = model.step(image_list[timestep], instruction)
                    
                    for elem in action:
                        action[elem] = action[elem].tolist()
                        
                    model_results["actions"].append({
                        "timestep": timestep,
                        "raw_action": raw_action.tolist(),
                        "action": action,
                        "generated_text": generated_text
                    })
                    
                    print(model_results["actions"][-1])
                    print('\n\n\n')
                
                end_time = time.time()
                model_results["time_taken"] = end_time - start_time

                # Store results in the results dictionary under the current folder and model name
                
                
                
                results[folder][model_text_name] = model_results

                # Save the updated results to the JSON file after processing each model
                with open(results_file, 'w') as json_file:
                    json.dump(results, json_file, indent=4)

if __name__ == '__main__':
    os.environ["DISPLAY"] = ":1" 
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    gpus = tf.config.list_physical_devices("GPU")
    if len(gpus) > 0:
        tf.config.set_logical_device_configuration(
            gpus[0],
            [tf.config.LogicalDeviceConfiguration(memory_limit=3072)],
        )
    main()
