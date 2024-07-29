"""
Simple script for real-to-sim eval using the prepackaged visual matching setup in ManiSkill2.
Example:
    cd {path_to_simpler_env_repo_root}
    python simpler_env/simple_inference_visual_matching_prepackaged_envs.py --policy rt1 \
        --ckpt-path ./checkpoints/rt_1_tf_trained_for_000400120  --task google_robot_pick_coke_can  --logging-root ./results_simple_eval/  --n-trajs 10
    python simpler_env/simple_inference_visual_matching_prepackaged_envs.py --policy octo-small \
        --ckpt-path None --task widowx_spoon_on_towel  --logging-root ./results_simple_eval/  --n-trajs 10
"""

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

parser = argparse.ArgumentParser()

# parser.add_argument("--policy", default="rt1", choices=["rt1", "octo-base", "octo-small", "vla"])
# parser.add_argument(
#     "--ckpt-path",
#     type=str,
#     default="./checkpoints/rt_1_x_tf_trained_for_002272480_step/",
# )

parser.add_argument("--logging-root", type=str, default="./results_simple_random_eval")
parser.add_argument("--tf-memory-limit", type=int, default=3072)
parser.add_argument("--tj", type=int, default=0)
parser.add_argument("--model", type=str, default="ECoT")

args = parser.parse_args()

# os.environ["DISPLAY"] = ""
os.environ["DISPLAY"] = ":1" 
# prevent a single jax process from taking up all the GPU memory
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
gpus = tf.config.list_physical_devices("GPU")
if len(gpus) > 0:
    # prevent a single tf process from taking up all the GPU memory
    tf.config.set_logical_device_configuration(
        gpus[0],
        [tf.config.LogicalDeviceConfiguration(memory_limit=args.tf_memory_limit)],
    )

tj = args.tj

images_path = "/iliad/group/datasets/bridgedata_v2/raw/bridge_data_v2/datacol1_toykitchen1/many_skills/0/2023-03-15_14-35-28/raw/traj_group0/traj" +str(tj) + "/images0"
file_path = "/iliad/group/datasets/bridgedata_v2/raw/bridge_data_v2/datacol1_toykitchen1/many_skills/0/2023-03-15_14-35-28/raw/traj_group0/traj" +str(tj) + "/lang.txt"

image_list = []
for filename in sorted(os.listdir(images_path), key=extract_number):
    if filename.endswith(".jpg"):
        img_path = os.path.join(images_path, filename)
        img = Image.open(img_path).convert('RGB')  # Ensure image is in RGB format
        img_array = np.array(img, dtype=np.uint8)
        image_list.append(img_array)

# run inference
from simpler_env.policies.openvla.vla_model import OpenVLAInference

policy_setup = "widowx_bridge"

# model = OpenVLAInference(saved_model_path=args.ckpt_path, policy_setup=policy_setup)
model = OpenVLAInference(model=args.model, policy_setup=policy_setup)


with open(file_path, 'r') as file:
    instruction = file.readline().strip()

model.reset(instruction)
print(instruction)

print(instruction)

images = []
predicted_terminated, success, truncated = False, False, False
timestep = 0
for timestep in range(len(image_list)):

    image = image_list[timestep]

    image = resize_image(image, (256, 256))

    raw_action, action, generated_text = model.step(image_list[timestep], instruction)

    if args.model == "ECoT":
        generate_reasoning_image(generated_text, image, timestep)

    # breakpoint()
    print(timestep)
    print(raw_action)

    # with open('simpler_env/test_data/ecot9vr.txt', 'a') as f:  # Use 'a' mode to append to the file
    #     np.savetxt(f, raw_action, fmt='%f', newline=' ')
    #     f.write('\n')  # Add a newline after each array

    images.append(image)
