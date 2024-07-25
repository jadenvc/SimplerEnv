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
from simpler_env.utils.env.observation_utils import get_image_from_maniskill2_obs_dict

from simpler_env_client import GymClient

from reasoning_utils import generate_reasoning_image, save_video_with_imageio

parser = argparse.ArgumentParser()

parser.add_argument("--policy", default="rt1", choices=["rt1", "octo-base", "octo-small", "vla"])
parser.add_argument(
    "--ckpt-path",
    type=str,
    default="./checkpoints/rt_1_x_tf_trained_for_002272480_step/",
)
parser.add_argument(
    "--task",
    default="google_robot_pick_horizontal_coke_can",
    choices=ENVIRONMENTS,
)
parser.add_argument("--logging-root", type=str, default="./results_simple_random_eval")
parser.add_argument("--tf-memory-limit", type=int, default=3072)
parser.add_argument("--n-trajs", type=int, default=10)

args = parser.parse_args()
if args.policy in ["octo-base", "octo-small"]:
    if args.ckpt_path in [None, "None"] or "rt_1_x" in args.ckpt_path:
        args.ckpt_path = args.policy
if args.ckpt_path[-1] == "/":
    args.ckpt_path = args.ckpt_path[:-1]
# logging_dir = os.path.join(args.logging_root, args.task, args.policy, os.path.basename(args.ckpt_path))
logging_dir = os.path.join(args.logging_root, args.task, args.policy, "ecot")
os.makedirs(logging_dir, exist_ok=True)

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

# client = GymClient("http://sadigh-ws-3.stanford.edu:5000")

# build environment
# env = simpler_env.make(args.task)
# client.make(args.task)

print("built env")

def extract_number(filename):
    # Assuming the filename format is img_#.jpg
    return int(filename.split('_')[1].split('.')[0])

def resize_image(img, resize_size):
    """Takes numpy array corresponding to a single image and returns resized image as numpy array."""
    assert isinstance(resize_size, tuple)
    img = Image.fromarray(img)
    BRIDGE_ORIG_IMG_SIZE = (256, 256)
    img = img.resize(BRIDGE_ORIG_IMG_SIZE, Image.Resampling.LANCZOS)
    img = img.resize(resize_size, Image.Resampling.LANCZOS)  # also resize to size seen at train time
    img = img.convert("RGB")
    img = np.array(img)
    return img

tj = 9

images_path = "/iliad/group/datasets/bridgedata_v2/raw/bridge_data_v2/datacol1_toykitchen1/many_skills/0/2023-03-15_14-35-28/raw/traj_group0/traj" +str(tj) + "/images0"
file_path = "/iliad/group/datasets/bridgedata_v2/raw/bridge_data_v2/datacol1_toykitchen1/many_skills/0/2023-03-15_14-35-28/raw/traj_group0/traj" +str(tj) + "/lang.txt"

image_list = []
for filename in sorted(os.listdir(images_path), key=extract_number):
    if filename.endswith(".jpg"):
        img_path = os.path.join(images_path, filename)
        img = Image.open(img_path).convert('RGB')  # Ensure image is in RGB format
        img_array = np.array(img, dtype=np.uint8)
        image_list.append(img_array)

# build policy
if "google_robot" in args.task:
    policy_setup = "google_robot"
elif "widowx" in args.task:
    policy_setup = "widowx_bridge"
else:
    raise NotImplementedError()

if args.policy == "rt1":
    from simpler_env.policies.rt1.rt1_model import RT1Inference

    model = RT1Inference(saved_model_path=args.ckpt_path, policy_setup=policy_setup)
elif "octo" in args.policy:
    from simpler_env.policies.octo.octo_model import OctoInference

    model = OctoInference(model_type=args.ckpt_path, policy_setup=policy_setup, init_rng=0)
else:
    from simpler_env.policies.openvla.vla_model import OpenVLAInference

    model = OpenVLAInference(saved_model_path=args.ckpt_path, policy_setup=policy_setup)


# run inference
success_arr = []
for ep_id in range(args.n_trajs):
    # image = client.reset() # what is this for?
    # instruction = client.get_language_instruction()
    # obs, reset_info = env.reset()
    # instruction = env.get_language_instruction()
    # for long-horizon environments, we check if the current subtask is the final subtask
    # is_final_subtask = client.is_final_subtask() 
    # is_final_subtask = env.is_final_subtask() 

    with open(file_path, 'r') as file:
        instruction = file.readline().strip()

    model.reset(instruction)
    print(instruction)

    
    # use client here as well
    # image = get_image_from_maniskill2_obs_dict(env, obs)  # np.ndarray of shape (H, W, 3), uint8
    # image = client.get_curr_image()


    # images = [image]
    images = []
    predicted_terminated, success, truncated = False, False, False
    timestep = 0
    while not (predicted_terminated or truncated):
        # step the model; "raw_action" is raw model action output; "action" is the processed action to be sent into maniskill env
        # breakpoint()

        # instruction = "move the green object in the silver bowl" # traj group 0
        # instruction = "take the blue spoon off the burner and put it on the right side of the burner"

        image = image_list[timestep]

        image = resize_image(image, (256, 256))

        raw_action, action, generated_text = model.step(image_list[timestep], instruction)
    

        # raw_action, action, generated_text = model.step(image, instruction)

        # breakpoint()

        generate_reasoning_image(generated_text, image, timestep)

        # predicted_terminated = bool(action["terminate_episode"][0] > 0)
        # if predicted_terminated:
        if timestep > 100:
            if not is_final_subtask:
                # advance the environment to the next subtask
                predicted_terminated = False
                # client.advance_to_next_subtask()


        # breakpoint()
        print(timestep)
        print(raw_action)

        with open('simpler_env/test_data/ecot9.txt', 'a') as f:  # Use 'a' mode to append to the file
            np.savetxt(f, raw_action, fmt='%f', newline=' ')
            f.write('\n')  # Add a newline after each array

        # image, reward, success, truncated, info = client.step(
        #     np.concatenate([action["world_vector"], action["rot_axangle"],np.array([action["gripper"]])]),
        # )
        # image, reward, success, truncated, info = client.step(raw_action)

        # print(raw_action)
        # obs, reward, success, truncated, info = env.step(
        #     np.concatenate([action["world_vector"], action["rot_axangle"], action["gripper"]]),
        # )
        # print(timestep, info)
        # new_instruction = client.get_language_instruction()

        # new_instruction = ""
        # # new_instruction = env.get_language_instruction()
        # if new_instruction != instruction:
        #     # update instruction for long horizon tasks
        #     instruction = new_instruction
        #     # print(instruction)

        # is_final_subtask = client.is_final_subtask()   
        # is_final_subtask = env.is_final_subtask() 
        # update image observation
        # image = get_image_from_maniskill2_obs_dict(env, obs)
        # images = client.get_curr_image(obs)  # update image observation
        # image = images[-1] if images else None
        images.append(image)
        # print(image)
        # if timestep > 10:
        #     media.write_video(f"{logging_dir}/episode_{ep_id}_success_{success}.mp4", images, fps=5)
        timestep += 1

    # episode_stats = info.get("episode_stats", {})
    success_arr.append(success)
    print(f"Episode {ep_id} success: {success}")
    save_path = f"{logging_dir}/episode_{ep_id}_success_{success}.mp4"
    save_video_with_imageio(images, save_path)
    # media.write_video(f"{logging_dir}/episode_{ep_id}_success_{success}.mp4", images, fps=5)

print(
    "**Overall Success**",
    np.mean(success_arr),
    f"({np.sum(success_arr)}/{len(success_arr)})",
)
