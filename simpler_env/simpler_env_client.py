import requests
import numpy as np
from PIL import Image
import gzip
from io import BytesIO
# import timer
from time import time
import base64


class GymClient:
    def __init__(self, base_url):
        self.base_url = base_url

    def make(self, task):
        response = requests.post(f"{self.base_url}/make", json={"task": task})
        return response.json()

    def reset(self):
        response = requests.post(f"{self.base_url}/reset")
        data = response.json()
        # breakpoint()
        image = self.decompress_image(data['image']).reshape(480,640,3)
        # image = np.frombuffer(base64.b64decode(data['image']), dtype=np.uint8).reshape(512, 640, 3)
        # reset_info = data['reset_info']
        return image

    def is_final_subtask(self):
        response = requests.post(f"{self.base_url}/is_final_subtask")
        return response.json()['is_final_subtask']

    def get_language_instruction(self):
        response = requests.post(f"{self.base_url}/get_language_instruction")
        return response.json()['language_instruction']

    def step(self, action):
        # print(action)
        response = requests.post(f"{self.base_url}/step", json={"action": action.tolist()})
        data = response.json()
        # breakpoint()
        image = self.decompress_image(data['image']).reshape(480,640,3)
        # image = np.frombuffer(base64.b64decode(data['image']), dtype=np.uint8).reshape(512, 640, 3)
        reward = data['reward']
        success = data["success"]
        truncated = data['truncated']
        info = data['info']
        return image, reward, success, truncated, info

    def advance_to_next_subtask(self):
        response = requests.post(f"{self.base_url}/advance_to_next_subtask")
        return response.json()
    

    def decompress_image(self, image_base64):
        compressed_img = base64.b64decode(image_base64)
        img_byte_array = gzip.decompress(compressed_img)
        # img = Image.open(BytesIO(img_byte_array))
        return np.frombuffer(img_byte_array, np.uint8)


# client = GymClient("http://sadigh-ws-3.stanford.edu:5000")
# # breakpoint()
# client.make("google_robot_pick_horizontal_coke_can")
# # start timer using time
# start = time()
# image = client.reset()
# for i in range(10):
#     action = np.zeros(7)
#     client.step(action)
#     # print time since start
#     print(time() - start)
#     print(client.get_language_instruction())