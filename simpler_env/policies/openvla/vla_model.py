from collections import defaultdict
from typing import Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import tf_agents
from tf_agents.policies import py_tf_eager_policy
from tf_agents.trajectories import time_step as ts
from transforms3d.euler import euler2axangle
from transformers import AutoProcessor, AutoModelForVision2Seq
import torch
import requests
import json_numpy
import base64
from PIL import Image
from pathlib import Path
from typing import Any, Dict, Optional, Union

# === Utilities ===
SYSTEM_PROMPT = (
    "A chat between a curious user and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the user's questions."
)


def get_openvla_prompt(instruction: str, openvla_path: Union[str, Path]) -> str:
    if True: #"v01" in openvla_path:
        # return f"{SYSTEM_PROMPT} USER: What action should the robot take to {instruction.lower()}? ASSISTANT:"
        return f"{SYSTEM_PROMPT} USER: What action should the robot take to {instruction.lower()}? ASSISTANT: TASK:"

    # else:
    #     return f"In: What action should the robot take to {instruction.lower()}?\nOut:"

def deserialize_numpy_array(serialized_dict):
    # Extract data from the dictionary
    encoded_data = serialized_dict['__numpy__']
    dtype = np.dtype(serialized_dict['dtype'])
    shape = tuple(serialized_dict['shape'])
    
    # Decode the base64 string
    decoded_data = base64.b64decode(encoded_data)
    
    # Convert the bytes back to a NumPy array
    array = np.frombuffer(decoded_data, dtype=dtype).reshape(shape)
    
    return array


class OpenVLAInference:
    def __init__(
        self,
        model: str = "ECoT",
        lang_embed_model_path: str = "https://tfhub.dev/google/universal-sentence-encoder-large/5",
        image_width: int = 320,
        image_height: int = 256,
        action_scale: float = 1.0,
        policy_setup: str = "google_robot",
    ) -> None:
        self.lang_embed_model = hub.load(lang_embed_model_path)
        # self.tfa_policy = py_tf_eager_policy.SavedModelPyTFEagerPolicy(
        #     model_path=saved_model_path,
        #     load_specs_from_pbtxt=True,
        #     use_tf_function=True,
        # )

        self.model = model

        self.device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

        if self.model == "ECoT":
            self.hf_model_path = "Embodied-CoT/ecot-openvla-7b-bridge"
            print("using ECoT")
        else:
            self.hf_model_path = "openvla/openvla-7b" 
            print("using open vla")
        
        self.processor = AutoProcessor.from_pretrained(self.hf_model_path, trust_remote_code=True)
        self.vla = AutoModelForVision2Seq.from_pretrained(
            self.hf_model_path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        ).to(self.device)

        # print("loaded vla")


        self.image_width = image_width
        self.image_height = image_height
        self.action_scale = action_scale

        self.observation = None
        self.tfa_time_step = None
        self.policy_state = None
        self.task_description = None
        self.task_description_embedding = None

        self.policy_setup = policy_setup
        if self.policy_setup == "google_robot":
            self.unnormalize_action = False
            self.unnormalize_action_fxn = None
            self.invert_gripper_action = False
            self.action_rotation_mode = "axis_angle"
        elif self.policy_setup == "widowx_bridge":
            self.unnormalize_action = True
            self.unnormalize_action_fxn = self._unnormalize_action_widowx_bridge
            self.invert_gripper_action = True
            self.action_rotation_mode = "rpy"
        else:
            raise NotImplementedError()

    @staticmethod
    def _rescale_action_with_bound(
        actions: np.ndarray | tf.Tensor,
        low: float,
        high: float,
        safety_margin: float = 0.0,
        post_scaling_max: float = 1.0,
        post_scaling_min: float = -1.0,
    ) -> np.ndarray:
        """Formula taken from https://stats.stackexchange.com/questions/281162/scale-a-number-between-a-range."""
        resc_actions = (actions - low) / (high - low) * (post_scaling_max - post_scaling_min) + post_scaling_min
        return np.clip(
            resc_actions,
            post_scaling_min + safety_margin,
            post_scaling_max - safety_margin,
        )

    def _unnormalize_action_widowx_bridge(self, action: dict[str, np.ndarray | tf.Tensor]) -> dict[str, np.ndarray]:
        action["world_vector"] = self._rescale_action_with_bound(
            action["world_vector"],
            low=-1.75,
            high=1.75,
            post_scaling_max=0.05,
            post_scaling_min=-0.05,
        )
        action["rotation_delta"] = self._rescale_action_with_bound(
            action["rotation_delta"],
            low=-1.4,
            high=1.4,
            post_scaling_max=0.25,
            post_scaling_min=-0.25,
        )
        return action

    def _initialize_model(self) -> None:
        # Perform one step of inference using dummy input to trace the tensoflow graph
        # Obtain a dummy observation, where the features are all 0
        # self.observation = tf_agents.specs.zero_spec_nest(
        #     tf_agents.specs.from_spec(self.tfa_policy.time_step_spec.observation)
        # )  # "natural_language_embedding": [512], "image", [256,320,3], "natural_language_instruction": <tf.Tensor: shape=(), dtype=string, numpy=b''>
        # # Construct a tf_agents time_step from the dummy observation
        # self.tfa_time_step = ts.transition(self.observation, reward=np.zeros((), dtype=np.float32))
        # # Initialize the state of the policy
        # self.policy_state = self.tfa_policy.get_initial_state(batch_size=1)
        # # Run inference using the policy
        # _action = self.tfa_policy.action(self.tfa_time_step, self.policy_state)
        self.policy_state = None

    def _resize_image(self, image: np.ndarray | tf.Tensor) -> tf.Tensor:
        image = tf.image.resize_with_pad(image, target_width=self.image_width, target_height=self.image_height)
        image = tf.cast(image, tf.uint8)
        return image

    def _initialize_task_description(self, task_description: Optional[str] = None) -> None:
        if task_description is not None:
            self.task_description = task_description
            self.task_description_embedding = self.lang_embed_model([task_description])[0]
        else:
            self.task_description = ""
            self.task_description_embedding = tf.zeros((512,), dtype=tf.float32)

    def reset(self, task_description: str) -> None:
        self._initialize_model()
        self._initialize_task_description(task_description)

    @staticmethod
    def _small_action_filter_google_robot(raw_action: dict[str, np.ndarray | tf.Tensor], arm_movement: bool = False, gripper: bool = True) -> dict[str, np.ndarray | tf.Tensor]:
        # small action filtering for google robot
        if arm_movement:
            raw_action["world_vector"] = tf.where(
                tf.abs(raw_action["world_vector"]) < 5e-3,
                tf.zeros_like(raw_action["world_vector"]),
                raw_action["world_vector"],
            )
            raw_action["rotation_delta"] = tf.where(
                tf.abs(raw_action["rotation_delta"]) < 5e-3,
                tf.zeros_like(raw_action["rotation_delta"]),
                raw_action["rotation_delta"],
            )
            raw_action["base_displacement_vector"] = tf.where(
                raw_action["base_displacement_vector"] < 5e-3,
                tf.zeros_like(raw_action["base_displacement_vector"]),
                raw_action["base_displacement_vector"],
            )
            raw_action["base_displacement_vertical_rotation"] = tf.where(
                raw_action["base_displacement_vertical_rotation"] < 1e-2,
                tf.zeros_like(raw_action["base_displacement_vertical_rotation"]),
                raw_action["base_displacement_vertical_rotation"],
            )
        if gripper:
            raw_action["gripper_closedness_action"] = tf.where(
                tf.abs(raw_action["gripper_closedness_action"]) < 1e-2,
                tf.zeros_like(raw_action["gripper_closedness_action"]),
                raw_action["gripper_closedness_action"],
            )
        return raw_action

    def step(self, image: np.ndarray, task_description: Optional[str] = None) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        """
        Input:
            image: np.ndarray of shape (H, W, 3), uint8
            task_description: Optional[str], task description; if different from previous task description, policy state is reset
        Output:
            raw_action: dict; raw policy action output
            action: dict; processed action to be sent to the maniskill2 environment, with the following keys:
                - 'world_vector': np.ndarray of shape (3,), xyz translation of robot end-effector
                - 'rot_axangle': np.ndarray of shape (3,), axis-angle representation of end-effector rotation
                - 'gripper': np.ndarray of shape (1,), gripper action
                - 'terminate_episode': np.ndarray of shape (1,), 1 if episode should be terminated, 0 otherwise
        """
        if task_description is not None:
            if task_description != self.task_description:
                # task description has changed; update language embedding
                # self._initialize_task_description(task_description)
                self.reset(task_description)
        
        assert image.dtype == np.uint8
        image = self._resize_image(image)
        # self.observation["image"] = image
        # self.observation["natural_language_embedding"] = self.task_description_embedding

        # inputs = processor(prompt, image).to(device, dtype=torch.bfloat16)

        # # Run OpenVLA Inference
        # start_time = time.time()

        # torch.manual_seed(0)
        # action, generated_ids = vla.predict_action(**inputs, unnorm_key="bridge_orig", do_sample=False, max_new_tokens=1024)
        # generated_text = processor.batch_decode(generated_ids)[0]

        torch.manual_seed(0)


        # breakpoint()

        image = np.array(image, dtype=np.uint8)
        instruction = self.task_description

        prompt = get_openvla_prompt(instruction, self.hf_model_path)

        inputs = self.processor(prompt, Image.fromarray(image).convert("RGB")).to(self.device, dtype=torch.bfloat16)
        # action = self.vla.predict_action(**inputs, unnorm_key=unnorm_key, do_sample=False)
        # generated_text = ""
        torch.manual_seed(0)
        # breakpoint()
        if self.model == "ECoT":
            raw_action, generated_ids = self.vla.predict_action(**inputs, unnorm_key="bridge_orig", do_sample=False, max_new_tokens=1024)
            # action, generated_ids = self.vla_pa(**inputs, unnorm_key="bridge_orig", do_sample=False, max_new_tokens=1024)
            generated_text = self.processor.batch_decode(generated_ids)[0]
        else:
            raw_action = self.vla.predict_action(**inputs, unnorm_key="bridge_orig", do_sample=False)
            generated_text = ""
        # breakpoint()
        # print("action")
        # print(raw_action)





        # payload = {
        #     "image": np.array(image).tolist(),
        #     "instruction": self.task_description
        # }

        # # action = requests.post("http://0.0.0.0:8000/act", json=payload).json()
        # response = requests.post("http://localhost:8000/act", json=payload).json()

        # # print("RESPONSE: ", response)

        # # breakpoint()

        # generated_text = response.get("generated_text", "")
        # # generated_text=""
        # action = response.get("action", {})  

        # raw_action = deserialize_numpy_array(action)




        # breakpoint()

        # print(raw_action)
        

        # inputs = self.processor(self.task_description, image, return_tensors="pt").to(self.device)
        # with torch.no_grad():
        #     action, generated_ids = self.model.predict_action(**inputs, unnorm_key="bridge_orig", do_sample=False, max_new_tokens=1024)

        # Process the raw action to fit the expected format
        formatted = {
            "world_vector": raw_action[:3],
            "rotation_delta": raw_action[3:6],
            "gripper_closedness_action": raw_action[6],
            # "terminate_episode": action[7]
        }

        return raw_action, formatted, generated_text

    def visualize_epoch(self, predicted_raw_actions: Sequence[np.ndarray], images: Sequence[np.ndarray], save_path: str) -> None:
        images = [self._resize_image(image) for image in images]
        predicted_action_name_to_values_over_time = defaultdict(list)
        figure_layout = [
            "terminate_episode_0",
            "terminate_episode_1",
            "terminate_episode_2",
            "world_vector_0",
            "world_vector_1",
            "world_vector_2",
            "rotation_delta_0",
            "rotation_delta_1",
            "rotation_delta_2",
            "gripper_closedness_action_0",
        ]
        action_order = [
            "terminate_episode",
            "world_vector",
            "rotation_delta",
            "gripper_closedness_action",
        ]

        for i, action in enumerate(predicted_raw_actions):
            for action_name in action_order:
                for action_sub_dimension in range(action[action_name].shape[0]):
                    # print(action_name, action_sub_dimension)
                    title = f"{action_name}_{action_sub_dimension}"
                    predicted_action_name_to_values_over_time[title].append(
                        predicted_raw_actions[i][action_name][action_sub_dimension]
                    )

        figure_layout = [["image"] * len(figure_layout), figure_layout]

        plt.rcParams.update({"font.size": 12})

        stacked = tf.concat(tf.unstack(images[::3], axis=0), 1)

        fig, axs = plt.subplot_mosaic(figure_layout)
        fig.set_size_inches([45, 10])

        for i, (k, v) in enumerate(predicted_action_name_to_values_over_time.items()):
            axs[k].plot(predicted_action_name_to_values_over_time[k], label="predicted action")
            axs[k].set_title(k)
            axs[k].set_xlabel("Time in one episode")

        axs["image"].imshow(stacked.numpy())
        axs["image"].set_xlabel("Time in one episode (subsampled)")

        plt.legend()
        plt.savefig(save_path)
