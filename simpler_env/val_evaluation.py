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
import tensorflow_datasets as tfds

from pprint import pprint


# # Make function that check if a folder exists, and if does not creates it example input 'eval_results/{i}.json'
# def check_folder_exists(folder):
#     if not os.path.exists(folder):
#         os.makedirs(folder)
        

# def main():
#     tf.random.set_seed(42)
#     ds_val = tfds.load('bridge_dataset', data_dir="/iliad/group/datasets/OXE_OCTO", split=f"val")
    
#     # Shuffle the dataset and take a subset
#     subset_size = 50
#     ds_subset = ds_val.shuffle(buffer_size=10000, seed=42).take(subset_size)
    
#     # print(sorted_folders)
#     results_dir = "val_eval_results"
#     check_folder_exists(results_dir)
    
#     for outeri in range(10):
#         # Load existing data if the file exists
#         results_file = f'{results_dir}/{i}.json'
        
#         try:
#             with open(results_file, 'r') as f:
#                 results = json.load(f)
#         except (json.JSONDecodeError, FileNotFoundError):
#             # Handle the error or initialize an empty dictionary
#             print("Error reading JSON file. Initializing an empty results dictionary.")
#             results = {}

#         for i, episode in enumerate(ds_subset):
#             episode_metadata = episode["episode_metadata"]
#             episode_name = f"traj{i}_{episode_metadata['episode_id']}"
            
#             if episode_metadata['has_language'].numpy() \
#                 and (episode_name not in results or len(list(results[episode_name].keys())) < 2): # Process only new folders
                    
#                 # traj_dir = os.path.join(TRAJ_DIR, folder)
#                 # images_path = os.path.join(traj_dir, "images0")
#                 # file_path = os.path.join(traj_dir, "lang.txt")
                
#                 print(f"Processing {episode_name}")
#                 # Check if lang.txt file exists
                
                
#                 image_list = []
                
#                 print("Collecting images and instructions")
#                 for step in episode["steps"]:
#                     image_list.append(step['observation']["image_0"].numpy())
#                     instruction = step['language_instruction'].numpy().decode('utf-8')  # Decode the instruction to a string


#                 for model_name in ["vla", "ECoT"]:
#                     start_time = time.time()
                    
#                     model_text_name = "ECoT" if model_name == "ECoT" else "OpenVLA"
#                     model = OpenVLAInference(model=model_name, policy_setup="widowx_bridge")
                    
#                     if episode_name not in results:
#                         results[episode_name] = {}
                        
#                     if len(list(results[episode_name].keys())) == 2:
#                         print("Already finished collecting the data for this setup")
#                         continue
                    
#                     # with open(file_path, 'r') as file:
#                     #     instruction = file.readline().strip()
                    
#                     model.reset(instruction)
                    
#                     model_results = {
#                         "instruction": instruction,
#                         "actions": [],
#                         "time_taken": 0
#                     }
                    
#                     predicted_terminated, success, truncated = False, False, False
                    
#                     for timestep in range(len(image_list)):
#                         image = image_list[timestep]
#                         image = resize_image(image, (256, 256))
#                         raw_action, action, generated_text = model.step(image_list[timestep], instruction)
                        
#                         for elem in action:
#                             action[elem] = action[elem].tolist()
                            
#                         model_results["actions"].append({
#                             "timestep": timestep,
#                             "raw_action": raw_action.tolist(),
#                             "action": action,
#                             "generated_text": generated_text
#                         })
                        
#                         print(model_results["actions"][-1])
#                         print('\n\n\n')
                    
#                     end_time = time.time()
#                     model_results["time_taken"] = end_time - start_time

#                     # Store results in the results dictionary under the current folder and model name
                    
                    
                    
#                     results[episode_name][model_text_name] = model_results

#                     # Save the updated results to the JSON file after processing each model
#                     with open(results_file, 'w') as json_file:
#                         json.dump(results, json_file, indent=4)

# if __name__ == '__main__':
#     os.environ["DISPLAY"] = ":1" 
#     os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
#     gpus = tf.config.list_physical_devices("GPU")
    # if len(gpus) > 0:
    #     tf.config.set_logical_device_configuration(
    #         gpus[0],
    #         [tf.config.LogicalDeviceConfiguration(memory_limit=3072)],
    #     )
    
    
#     main()


# import json
# import os
# import time
# import tensorflow as tf
# import tensorflow_datasets as tfds
# from simpler_env.policies.openvla.vla_model import OpenVLAInference
# from reasoning_utils import resize_image

def check_folder_exists(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)

def main():
    tf.random.set_seed(42)

    # Load the dataset
    ds_val = tfds.load('bridge_dataset', data_dir="/iliad/group/datasets/OXE_OCTO", split='val')

    # Shuffle with a controlled buffer and take the subset
    subset_size = 50
    ds_subset = ds_val.shuffle(buffer_size=10000, seed=42, reshuffle_each_iteration=False).take(subset_size)


    results_dir = "val_eval_results"
    check_folder_exists(results_dir)
    
    episode_usage = []
    for idx, interim_episode in enumerate(ds_subset):
        preli_episode_metadata = interim_episode["episode_metadata"]
        preli_episode_name = f"traj{idx}_{preli_episode_metadata['episode_id']}"
        episode_usage.append(preli_episode_name)
    
    # Now write the episode_usage to a text file comma separated in the results_dir
    with open(f"{results_dir}/episode_usage.txt", "a") as f:
        f.write('0,' + ",".join(episode_usage) + "\n")
        
        
    for outer_i in range(10):
        results_file = f'{results_dir}/{outer_i}.json'
        try:
            with open(results_file, 'r') as f:
                results = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            print("Error reading JSON file. Initializing an empty results dictionary.")
            results = {}
            
        episode_usage = []
        for idx, interim_episode in enumerate(ds_subset):
            preli_episode_metadata = interim_episode["episode_metadata"]
            preli_episode_name = f"traj{idx}_{preli_episode_metadata['episode_id']}"
            episode_usage.append(preli_episode_name)
        
        # Now write the episode_usage to a text file comma separated in the results_dir
        with open(f"{results_dir}/episode_usage.txt", "a") as f:
            f.write(f"{outer_i+1}," + ",".join(episode_usage) + "\n")

        for inner_i, episode in enumerate(ds_subset):
            episode_metadata = episode["episode_metadata"]
            episode_name = f"traj{inner_i}_{episode_metadata['episode_id']}"

            if episode_metadata['has_language'].numpy() and \
                (episode_name not in results or len(list(results[episode_name].keys())) < 2):

                print(f"Processing {episode_name}")
                image_list = []
                for step in episode["steps"]:
                    image_list.append(step['observation']["image_0"].numpy())
                    instruction = step['language_instruction'].numpy().decode('utf-8')

                for model_name in ["vla", "ECoT"]:
                    if episode_name not in results:
                        results[episode_name] = {}

                    if len(list(results[episode_name].keys())) == 2:
                        print("Already finished collecting the data for this setup")
                        continue

                    start_time = time.time()
                    model_text_name = "ECoT" if model_name == "ECoT" else "OpenVLA"
                    model = OpenVLAInference(model=model_name, policy_setup="widowx_bridge")
                    model.reset(instruction)

                    model_results = {
                        "instruction": instruction,
                        "actions": [],
                        "time_taken": 0
                    }

                    for timestep in range(len(image_list)):
                        # image = resize_image(image_list[timestep], (256, 256))
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
                    results[episode_name][model_text_name] = model_results

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
