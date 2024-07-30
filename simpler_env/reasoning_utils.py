import torch
from transformers import AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor

import time
import numpy as np
import cv2
import textwrap
from PIL import Image, ImageDraw, ImageFont
import enum

import requests
from io import BytesIO
from PIL import Image
import imageio

import os

def save_video_with_imageio(images, filename, fps=5):
    with imageio.get_writer(filename, fps=fps) as writer:
        for image in images:
            writer.append_data(image)

def split_reasoning(text, tags):
    new_parts = {None: text}

    for tag in tags:
        parts = new_parts
        new_parts = dict()

        for k, v in parts.items():
            if tag in v:
                s = v.split(tag)
                new_parts[k] = s[0]
                new_parts[tag] = s[1]
                # print(tag, s)
            else:
                new_parts[k] = v

    return new_parts

class CotTag(enum.Enum):
    TASK = "TASK:"
    PLAN = "PLAN:"
    VISIBLE_OBJECTS = "VISIBLE OBJECTS:"
    SUBTASK_REASONING = "SUBTASK REASONING:"
    SUBTASK = "SUBTASK:"
    MOVE_REASONING = "MOVE REASONING:"
    MOVE = "MOVE:"
    GRIPPER_POSITION = "GRIPPER POSITION:"
    ACTION = "ACTION:"


def get_cot_tags_list():
    return [
        CotTag.TASK.value,
        CotTag.PLAN.value,
        CotTag.VISIBLE_OBJECTS.value,
        CotTag.SUBTASK_REASONING.value,
        CotTag.SUBTASK.value,
        CotTag.MOVE_REASONING.value,
        CotTag.MOVE.value,
        CotTag.GRIPPER_POSITION.value,
        CotTag.ACTION.value,
    ]

def name_to_random_color(name):
    return [(hash(name) // (256**i)) % 256 for i in range(3)]




def draw_gripper(img, pos_list, img_size=(336, 336)):
    for i, pos in enumerate(reversed(pos_list)):
        pos = resize_pos(pos, img_size)
        scale = 255 - int(255 * i / len(pos_list))
        cv2.circle(img, pos, 6, (0, 0, 0), -1)
        cv2.circle(img, pos, 5, (scale, scale, 255), -1)

def get_metadata(reasoning):
    metadata = {"gripper": [[0, 0]], "bboxes": dict()}

    if f" {CotTag.GRIPPER_POSITION.value}" in reasoning:
        gripper_pos = reasoning[f" {CotTag.GRIPPER_POSITION.value}"]
        gripper_pos = gripper_pos.split("[")[-1]
        gripper_pos = gripper_pos.split("]")[0]
        gripper_pos = [int(x.strip()) for x in gripper_pos.split(",") if x.strip()]
        gripper_pos = [(gripper_pos[2 * i], gripper_pos[2 * i + 1]) for i in range(len(gripper_pos) // 2)]
        metadata["gripper"] = gripper_pos

    if f" {CotTag.VISIBLE_OBJECTS.value}" in reasoning:
        for sample in reasoning[f" {CotTag.VISIBLE_OBJECTS.value}"].split("]"):
            obj = sample.split("[")[0].strip()
            if obj == "":
                continue
            coords = [int(n.strip()) for n in sample.split("[")[-1].split(",") if n.strip()]
            metadata["bboxes"][obj] = coords

    return metadata

def resize_pos(pos, img_size):
    return [(x * size) // 256 for x, size in zip(pos, img_size)]

def draw_bboxes(img, bboxes, img_size=(336, 336)):
    for name, bbox in bboxes.items():
        show_name = name
        # show_name = f'{name}; {str(bbox)}'

        cv2.rectangle(
            img,
            resize_pos((bbox[0], bbox[1]), img_size),
            resize_pos((bbox[2], bbox[3]), img_size),
            name_to_random_color(name),
            1,
        )
        cv2.putText(
            img,
            show_name,
            resize_pos((bbox[0], bbox[1] + 6), img_size),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

#@title Visualize reasoning and image
def generate_reasoning_image(generated_text, image, timestep):
    # print(generated_text)
    tags = [f" {tag}" for tag in get_cot_tags_list()]
    reasoning = split_reasoning(generated_text, tags)
    text = [tag + reasoning[tag] for tag in [' TASK:',' PLAN:',' SUBTASK REASONING:',' SUBTASK:',
                                            ' MOVE REASONING:',' MOVE:', ' VISIBLE OBJECTS:', ' GRIPPER POSITION:'] if tag in reasoning]
    metadata = get_metadata(reasoning)
    bboxes = {}
    for k, v in metadata["bboxes"].items():
        if k[0] == ",":
            k = k[1:]
        bboxes[k.lstrip().rstrip()] = v

    caption = ""
    for t in text:
        wrapper = textwrap.TextWrapper(width=80, replace_whitespace=False)
        word_list = wrapper.wrap(text=t)
        caption_new = ''
        for ii in word_list[:-1]:
            caption_new = caption_new + ii + '\n      '
        caption_new += word_list[-1]

        caption += caption_new.lstrip() + "\n\n"

    base = Image.fromarray(np.ones((256, 640, 3), dtype=np.uint8) * 255)
    draw = ImageDraw.Draw(base)
    font_path = os.path.join(cv2.__path__[0],'qt','fonts','DejaVuSans.ttf')
    font = ImageFont.truetype(font_path, size=14)
    color = (0,0,0) # RGB
    draw.text((30, 30), caption, color, font=font)

    img_arr = np.array(image)
    draw_gripper(img_arr, metadata["gripper"])
    draw_bboxes(img_arr, bboxes)

    text_arr = np.array(base)

    reasoning_img = Image.fromarray(np.concatenate([img_arr, text_arr], axis=1))

    output_dir = "/iliad/u/jvclark/SimplerEnv/results_simple_eval/test_reasonings"
    os.makedirs(output_dir, exist_ok=True)

    # Save the image
    output_path = os.path.join(output_dir, f"reasoning_{timestep}.png")
    reasoning_img.save(output_path)


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