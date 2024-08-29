import argparse
import os
import numpy as np
# import tensorflow as tf
# from simpler_env.policies.openvla.vla_model import OpenVLAInference
from reasoning_utils import generate_reasoning_image, resize_image
import json

parser = argparse.ArgumentParser()

parser.add_argument("--logging-root", type=str, default="./labels_demos")
parser.add_argument("--tf-memory-limit", type=int, default=3072)
parser.add_argument("--model", type=str, default="ECoT")
parser.add_argument("--data-dir", type=str, default="/iliad/u/jvclark/jaden")

args = parser.parse_args()

# os.environ["DISPLAY"] = ":1"
# os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
# gpus = tf.config.list_physical_devices("GPU")
# if len(gpus) > 0:
#     tf.config.set_logical_device_configuration(
#         gpus[0],
#         [tf.config.LogicalDeviceConfiguration(memory_limit=args.tf_memory_limit)],
#     )

# Load the json file
with open("/iliad/u/jvclark/action-reasoning/scripts/generate_embodied_data/reasoning_data/complete/new_reasonings_fr3_bridge_filtered.json", "r") as f:
    fr3d = json.load(f)

# Initialize the model
# policy_setup = "widowx_bridge"
# model = OpenVLAInference(model=args.model, policy_setup=policy_setup)

# List all .npz files in the directory
npz_files = [f for f in os.listdir(args.data_dir) if f.endswith('.npz')]

for key in list(fr3d.keys()):
    episode = fr3d[key]
    id = list(episode.keys())[0]
    episode_label = episode[id]


    # # Construct the full file path
    # file_path = os.path.join(args.data_dir, npz_file)

    # # Extract the demo index from the filename (e.g., "20240731T171656_1_374.npz" -> "1")
    # demo_index = npz_file.split('_')[1]

    # # Check if we need to move to the next instruction
    # if demo_index == "1" and current_instruction_index < len(instructions) - 1:
    #     current_instruction_index += 1

    # # Get the instruction text for the current demo
    # instruction_text = instructions[current_instruction_index]

    # Load the .npz file
    with np.load(key) as data:
        # Check if the key "obs.agent_image" exists
        if "obs.front_image" in data.files:
            images = data["obs.front_image"]

            # Convert images to a list of numpy arrays
            image_list = [np.transpose(image, (1, 2, 0)) for image in images[33:]]

            # Reset the model with the instruction
            # model.reset(instruction_text)
            print(f"Running inference for {key}")

            # Create a directory to save reasoning images
            try:
                reasoning_dir = os.path.join(args.logging_root, f"{key[:-4]}_{str(episode[id]['reasoning']['0']['task']).replace(' ', '_')}")
                # breakpoint()
                os.makedirs(reasoning_dir, exist_ok=True)
                id = list(episode.keys())[0]
                for timestep, image in enumerate(image_list):
                    if not timestep % 15 == 0:
                        continue
                    image = resize_image(image, (256, 256))
                    generated_text = str(episode[id]['reasoning'][str(timestep)])

                    # Get the model's action and reasoning
                    # raw_action, action, generated_text = model.step(image, instruction_text)

                    
                    generate_reasoning_image(generated_text, image, timestep, output_dir=reasoning_dir)

                    # Log and store the action (can be modified as needed)
                    print(f"Timestep: {timestep}")
            except:
                print(f"Failed to save reasoning images for {key}")
                continue

print("Finished saving reasoning images.")
