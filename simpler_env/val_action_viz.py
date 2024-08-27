import streamlit as st
import os
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import numpy as np
import pickle
import json
import cv2
import re
from textwrap import wrap
from tqdm.auto import tqdm
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import tensorflow_datasets as tfds
import tensorflow as tf
import seaborn as sns

ALREADY_PROCESSED = False
# Set wide mode
st.set_page_config(layout="wide")


# Labels for the 7 numbers
action_element_labels = ['WV 1', 'WV 2', 'WV 3', 'RD 1', 'RD 2', 'RD 3', 'Gripper Closedness']
x_action_elements = np.arange(len(action_element_labels))  # the label locations
width = 0.35  # the width of the bars

def load_multiple_results(num_files=8):
    all_results = []
    for i in range(num_files):
        file_path = f'val_100_eval_results/{i}.json'
        try:
            with open(file_path, 'r') as f:
                all_results.append(json.load(f))
        except (json.JSONDecodeError, FileNotFoundError):
            print(f"Error reading JSON file {file_path}. Skipping this file.")
    return all_results

def calculate_action_uncertainty(all_results, trajectory, frame_index, low, high):
    action_data = []
    # print(f"Calculating action uncertainty for frame {frame_index} in trajectory {trajectory}...")
    # print(len(all_results))
    
    for result in all_results:
        if trajectory in result and "ECoT" in result[trajectory]:
            action = result[trajectory]["ECoT"]['actions'][frame_index]["raw_action"]
            # print(action)
            normalized_action = min_max_normalize(np.array(action), low, high)
            action_data.append(normalized_action)
    
    # print(f"Action data for frame {frame_index} in trajectory {trajectory}: {action_data}")
    return np.array(action_data)

def create_uncertainty_boxplot(action_data):
    fig, ax = plt.subplots(figsize=(12, 6))
    if action_data.size == 0:
        print("Warning: Empty action data provided to boxplot.")
    sns.boxplot(data=action_data, ax=ax)
    ax.set_xticks(np.arange(len(action_element_labels)))
    ax.set_xticklabels(action_element_labels)
    ax.set_xlabel('Action Dimensions')
    ax.set_ylabel('Normalized Action Values')
    ax.set_title('Uncertainty Across Action Dimensions (Normalized)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    return fig



@st.cache_resource
def load_dataset():
    tf.random.set_seed(42)
    ds_val = tfds.load('bridge_dataset', data_dir="/iliad/group/datasets/OXE_OCTO", split='val')
    subset_size = 100
    return ds_val.shuffle(buffer_size=10000, seed=42, reshuffle_each_iteration=False).take(subset_size)

@st.cache_data
def load_results():
    results_file = 'val_100_eval_results/0.json'
    try:
        with open(results_file, 'r') as f:
            return json.load(f)
    except (json.JSONDecodeError, FileNotFoundError):
        st.error("Error reading JSON file. Initializing an empty results dictionary.")
        return {}

@st.cache_data
def calculate_mse(actions_gt, actions_model):
    return [np.mean((np.array(gt) - np.array(model)) ** 2) for gt, model in zip(actions_gt, actions_model)]

@st.cache_data
def process_actions(steps):
    return [step['action'] for step in steps]


def min_max_normalize(actions, low, high):
    """
    Normalize the actions using the formula: 2 * (action - low) / (high - low) - 1
    where low = q01 and high = q99.
    
    Parameters:
    actions (numpy.array or torch.Tensor): The actions to be normalized.
    low (numpy.array or torch.Tensor): The low (q01) values for each action dimension.
    high (numpy.array or torch.Tensor): The high (q99) values for each action dimension.
    
    Returns:
    normalized_actions (same type as input): The normalized actions.
    """
    return 2 * (actions - low) / (high - low) - 1


@st.cache_data
def compute_mse_dict(results, _ds_subset):
    mse_dict = {}
    
    # open and load this json /iliad/group/datasets/OXE_OCTO/bridge_dataset/1.0.0/dataset_statistics_0877b5ade4c695b2af4659ecbe4c9887e80f7d64fbb1401e76227352dbee94b9.json
    with open('/iliad/group/datasets/OXE_OCTO/bridge_dataset/1.0.0/dataset_statistics_0877b5ade4c695b2af4659ecbe4c9887e80f7d64fbb1401e76227352dbee94b9.json', 'r') as f:
        dataset_statistics = json.load(f)
        
    low = np.array(dataset_statistics['action']['q01'])
    high = np.array(dataset_statistics['action']['q99'])
    
    for i, trajectory_obj in enumerate(_ds_subset):
        trajectory_metadata = trajectory_obj["episode_metadata"]
        trajectory_steps = trajectory_obj["steps"]
        trajectory = f"traj{i}_{trajectory_metadata['episode_id']}"
        
        if trajectory_metadata['has_language'].numpy() and trajectory in results and len(list(results[trajectory].keys())) == 2:
            gt_actions = process_actions(trajectory_steps)
            vla_actions = [elem['raw_action'] for elem in results[trajectory]["OpenVLA"]['actions']]
            ecot_actions = [elem['raw_action'] for elem in results[trajectory]["ECoT"]['actions']]
            
            # Normalize the actions using min-max normalization
            gt_actions = min_max_normalize(np.array(gt_actions), low, high)
            vla_actions = min_max_normalize(np.array(vla_actions), low, high)
            ecot_actions = min_max_normalize(np.array(ecot_actions), low, high)

            
            mse_dict[trajectory] = {
                "VLA": calculate_mse(gt_actions, vla_actions),
                "ECOT": calculate_mse(gt_actions, ecot_actions)
            }
    return mse_dict

def extract_trajectory_info(trajectory):
    """
    Extracts the index `i` and `episode_id` from a trajectory string formatted as "traj{i}_{episode_id}".

    Parameters:
        trajectory (str): The trajectory string to extract the information from.

    Returns:
        tuple: A tuple containing `i` (int) and `episode_id` (int).
    """
    pattern = r"traj(\d+)_(\d+)"
    match = re.match(pattern, trajectory)
    
    if match:
        i = int(match.group(1))  # Extract and convert `i` to an integer
        episode_id = int(match.group(2))  # Extract and convert `episode_id` to an integer
        return i, episode_id
    else:
        raise ValueError("The trajectory string format did not match the expected pattern.")
    
    
def parse_reasoning_string(reasoning_string):
    # Extract gripper position
    gripper_position_match = re.search(r'GRIPPER POSITION:\s*\[([0-9.,\s-]+)\]', reasoning_string)
    if gripper_position_match:
        gripper_position = [int(coord) for coord in gripper_position_match.group(1).split(',')]
    else:
        gripper_position = None

    # Extract visible objects
    visible_objects_match = re.search(r'VISIBLE OBJECTS: (.*)$', reasoning_string, re.DOTALL)
    if visible_objects_match:
        visible_objects_str = visible_objects_match.group(1).strip()
        visible_objects = re.findall(r'([\w\s]+)\[(\d+,\s*\d+,\s*\d+,\s*\d+)\]', visible_objects_str)
        visible_objects = [(obj[0].strip(), [int(coord) for coord in obj[1].split(',')]) for obj in visible_objects]
    else:
        visible_objects = []

    # Remove gripper position and visible objects from reasoning string
    cleaned_reasoning_string = re.sub(r'GRIPPER POSITION:.*?VISIBLE OBJECTS:.*$', '', reasoning_string, flags=re.DOTALL).strip()

    return gripper_position, visible_objects, cleaned_reasoning_string


def create_composite_image(image_array, gripper_position, visible_objects, reasoning_text):
    # Load the image
    image = Image.fromarray(image_array)
    original_width, original_height = image.size

    # Check if the original image is 256x256 and scale it up if so
    image = image.resize((1024, 1024))
    original_width, original_height = image.size

    # Scaling factors (since the image is now scaled up to 1024x1024)
    scale_x = original_width / 256
    scale_y = original_height / 256
    
    # Create a new image with double the width to accommodate the text
    composite_image = Image.new('RGB', (2 * original_width, original_height), (255, 255, 255))
    composite_image.paste(image, (0, 0))

    # Draw on the image
    draw = ImageDraw.Draw(composite_image)

    # Draw gripper position (if available)
    if gripper_position:
        try:
            scaled_gripper_position = [int(gripper_position[0] * scale_x), int(gripper_position[1] * scale_y)]
            draw.ellipse(
                (scaled_gripper_position[0] - 5, scaled_gripper_position[1] - 5,
                 scaled_gripper_position[0] + 5, scaled_gripper_position[1] + 5),
                fill='red', outline='red'
            )
        except Exception as e:
            print(f"Error drawing gripper position: {e}. Skipping this part.")

    # Define font for text with increased size
    font1 = ImageFont.load_default().font_variant(size=44)  # Larger size for main text
    font2 = ImageFont.load_default().font_variant(size=32)
        
    # Draw visible objects bounding boxes and labels
    for obj_label, bbox in visible_objects:
        try:
            scaled_bbox = [int(coord * scale_x if i % 2 == 0 else coord * scale_y) for i, coord in enumerate(bbox)]

            # Ensure the coordinates are valid
            x0, y0, x1, y1 = scaled_bbox
            if x1 < x0:
                x0, x1 = x1, x0
            if y1 < y0:
                y0, y1 = y1, y0

            # Debugging information
            # print(f"Drawing rectangle for {obj_label} with coordinates: ({x0}, {y0}, {x1}, {y1})")

            draw.rectangle([x0, y0, x1, y1], outline='blue', width=2)
            draw.text((x0, y0 - 20), obj_label, fill='blue', font=font2)
        except Exception as e:
            print(f"Error drawing rectangle for {obj_label}: {e}. Skipping this part.")

    # Draw reasoning text on the right side with text wrapping
    try:
        text_x = original_width + 10
        text_y = 10
        max_width = original_width - 20  # Maximum width for text
        line_height = font1.size + 1  # Adjust line height based on font size

        for paragraph in reasoning_text.split('\n'):
            # Wrap text for each paragraph
            wrapped_lines = wrap(paragraph, width=int(max_width / (font1.size / 2)))  # Estimate characters per line
            for line in wrapped_lines:
                draw.text((text_x, text_y), line, fill='black', font=font1)
                text_y += line_height
            text_y += line_height // 2  # Add some extra space between paragraphs
    except Exception as e:
        print(f"Error drawing reasoning text: {e}. Skipping this part.")

    return composite_image



def plot_mse(ecot_single_action_mse, vla_single_action_mse):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x_action_elements - width/2, ecot_single_action_mse, width, label='ECoT', color='red')
    ax.bar(x_action_elements + width/2, vla_single_action_mse, width, label='OpenVLA', color='green')
    
    ax.set_xlabel('Action Element Label')
    ax.set_ylabel('Mean Squared Error')
    ax.set_title('MSE Comparison Between Two Model Outputs and Ground Truth')
    ax.set_xticks(x_action_elements)
    ax.set_xticklabels(action_element_labels)
    ax.grid(axis='y')
    ax.legend()
    
    return fig

def plot_action_deltas(deltas):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x_action_elements, deltas, color='coral')
    
    ax.set_xlabel('Action Element Label')
    ax.set_ylabel('Delta (Absolute Difference)')
    ax.set_title('Delta Between Two Model Predictions')
    ax.set_xticks(x_action_elements)
    ax.set_xticklabels(action_element_labels)
    ax.grid(axis='y')
    
    return fig

@st.cache_data
def format_printable_action(data):
    formatted_str = ""
    for key, value in data.items():
        formatted_str += f"{key}:\n\n"
        if isinstance(value, list):
            for number in value:
                formatted_str += f"  {number}\n\n"
        else:
            formatted_str += f"  {value}\n\n"
    return formatted_str

@st.cache_data
def format_reasoning_string(input_str):
    clean_input = input_str.replace("<s>", "").replace("</s>", "").strip()
    user_segment, assistant_segment = clean_input.split("ASSISTANT:", 1)
    user_question = user_segment.replace("USER:", "").strip()
    
    parts = ["PLAN:", "SUBTASK REASONING:", "SUBTASK:", "MOVE REASONING:", 
             "MOVE:", "GRIPPER POSITION:", "VISIBLE OBJECTS:"]
    
    formatted_response = []
    for part in parts:
        if part in assistant_segment:
            start_index = assistant_segment.index(part) + len(part)
            end_index = len(assistant_segment)
            for next_part in parts:
                next_index = assistant_segment.find(next_part, start_index)
                if next_index != -1 and next_index < end_index:
                    end_index = next_index
            content = assistant_segment[start_index:end_index].strip()
            formatted_response.append(f"\n{part} {content}")
    
    formatted_output = "\n".join(formatted_response)
    
    action_start_index = formatted_output.find('ACTION:')
    
    return formatted_output[:action_start_index]

def process_actions(steps):
    actions = []
    for step in steps:
        actions.append(step['action'])
    return actions


def calculate_mse(actions_gt, actions_model):
    mse_list = [np.mean((gt - model) ** 2) for gt, model in zip(actions_gt, actions_model)]
    return mse_list

@st.cache_data
def generate_and_store_visualizations(results, _ds_subset, output_directory):
    global ALREADY_PROCESSED
    if ALREADY_PROCESSED:
        return
    
    all_results = load_multiple_results()
    
    # open and load this json /iliad/group/datasets/OXE_OCTO/bridge_dataset/1.0.0/dataset_statistics_0877b5ade4c695b2af4659ecbe4c9887e80f7d64fbb1401e76227352dbee94b9.json
    with open('/iliad/group/datasets/OXE_OCTO/bridge_dataset/1.0.0/dataset_statistics_0877b5ade4c695b2af4659ecbe4c9887e80f7d64fbb1401e76227352dbee94b9.json', 'r') as f:
        dataset_statistics = json.load(f)
        
    low = np.array(dataset_statistics['action']['q01'])
    high = np.array(dataset_statistics['action']['q99'])
    
    os.makedirs(output_directory, exist_ok=True)
    
    for i, trajectory_obj in tqdm(enumerate(_ds_subset), total=len(_ds_subset)):
        
        trajectory_metadata = trajectory_obj["episode_metadata"]
        trajectory_steps = trajectory_obj["steps"]
        
        trajectory = f"traj{i}_{trajectory_metadata['episode_id']}"
        
        if trajectory_metadata['has_language'].numpy() and \
                (trajectory in results and len(list(results[trajectory].keys())) == 2):
                    
            # print(f"Generating visualizations for {trajectory}...")
            
            trajectory_dir = os.path.join(output_directory, trajectory)
            os.makedirs(trajectory_dir, exist_ok=True)
            
            # if results.get(trajectory) is None:
            #     print(f"Results not found for {trajectory}. Skipping visualization generation.")
            #     continue
            
            # if results[trajectory].get("OpenVLA") is None or results[trajectory].get("ECoT") is None:
            #     print(f"Results not found for {trajectory}. Skipping visualization generation.")
            #     continue
            
            
            
            gt_actions = process_actions(trajectory_steps)
            vla_actions = [elem['raw_action'] for elem in results[trajectory]["OpenVLA"]['actions']]
            ecot_actions = [elem['raw_action'] for elem in results[trajectory]["ECoT"]['actions']]
            
            # Normalize the actions using min-max normalization
            gt_actions = min_max_normalize(np.array(gt_actions), low, high)
            vla_actions = min_max_normalize(np.array(vla_actions), low, high)
            ecot_actions = min_max_normalize(np.array(ecot_actions), low, high)
            
            vla_mse = calculate_mse(gt_actions, vla_actions)
            ecot_mse = calculate_mse(gt_actions, ecot_actions)
            
            mse_dict[trajectory] = {
                
                "VLA": vla_mse,
                "ECOT": ecot_mse
            }
            
            # Generate and save MSE plot
            mse_plot_path = os.path.join(trajectory_dir, 'mse_plot.png')
            if not os.path.exists(mse_plot_path):
                plt.figure(figsize=(12, 6))
                plt.plot(vla_mse, label='VLA', color='green')
                plt.plot(ecot_mse, label='ECOT', color='red')
                plt.xlabel('Time Step')
                plt.ylabel('Mean Squared Error')
                plt.title(f'MSE over Time Steps for {trajectory}')
                plt.legend()
                plt.grid(True)
                plt.savefig(mse_plot_path)
                plt.close()
            
            # Generate and save frame-specific visualizations
            for frame_index, step in enumerate(trajectory_steps):
                frame_dir = os.path.join(trajectory_dir, f'frame_{frame_index}')
                os.makedirs(frame_dir, exist_ok=True)
                
                #before you go on to generate all the images, check if the below elements exist. If they do, just skip the current iteration
                if os.path.exists(os.path.join(frame_dir, 'composite_image.png')) \
                    and os.path.exists(os.path.join(frame_dir, 'mse_comparison.png')) \
                    and os.path.exists(os.path.join(frame_dir, 'action_deltas.png')) \
                    and os.path.exists(os.path.join(frame_dir, 'frame_mse_comparison.png'))\
                    and os.path.exists(os.path.join(frame_dir, 'uncertainty_boxplot.png')):
                    continue
                
                # Generate and save composite image
                composite_image_path = os.path.join(frame_dir, 'composite_image.png')
                if not os.path.exists(composite_image_path):
                    # frame_path = os.path.join(image_directory_base, f'{trajectory}/images0', f"im_{frame_index}.jpg")
                    frame_array = step['observation']['image_0'].numpy()
                    # Try printing len of trajecotry steps
                    # print(len(trajectory_steps))
                    # print(trajectory)
                    # print(len(results[trajectory]["ECoT"]['actions']))
                    # # print(results[trajectory]["ECoT"]['actions'])
                    # print(frame_index)
                    formatted_reasoning_string = format_reasoning_string(results[trajectory]["ECoT"]['actions'][frame_index]["generated_text"])
                    gripper_position, visible_objects, cleaned_reasoning_string = parse_reasoning_string(formatted_reasoning_string)
                    composite_image = create_composite_image(frame_array, gripper_position, visible_objects, cleaned_reasoning_string)
                    composite_image.save(composite_image_path)
                
                # Generate and save MSE comparison plot
                mse_comparison_path = os.path.join(frame_dir, 'mse_comparison.png')
                if not os.path.exists(mse_comparison_path):
                    ecot_single_action_mse = (np.array(results[trajectory]["ECoT"]['actions'][frame_index]["raw_action"]) - np.array(gt_actions[frame_index]))**2
                    vla_single_action_mse = (np.array(results[trajectory]["OpenVLA"]['actions'][frame_index]["raw_action"]) - np.array(gt_actions[frame_index]))**2
                    fig = plot_mse(ecot_single_action_mse, vla_single_action_mse)
                    fig.savefig(mse_comparison_path)
                    plt.close(fig)
                
                # Generate and save action deltas plot
                action_deltas_path = os.path.join(frame_dir, 'action_deltas.png')
                if not os.path.exists(action_deltas_path):
                    deltas = np.abs(np.array(results[trajectory]["ECoT"]['actions'][frame_index]["raw_action"]) - np.array(results[trajectory]["OpenVLA"]['actions'][frame_index]["raw_action"]))
                    fig = plot_action_deltas(deltas)
                    fig.savefig(action_deltas_path)
                    plt.close(fig)
                
                # Generate and save frame MSE comparison plot
                frame_mse_comparison_path = os.path.join(frame_dir, 'frame_mse_comparison.png')
                if not os.path.exists(frame_mse_comparison_path):
                    fig, ax = plt.subplots()
                    ax.bar(['VLA', 'ECOT'], [vla_mse[frame_index], ecot_mse[frame_index]], color=['green', 'red'])
                    ax.set_ylabel('Mean Squared Error')
                    ax.set_title('MSE Comparison for Frame')
                    fig.savefig(frame_mse_comparison_path)
                    plt.close(fig)
                    
                # Generate and save uncertainty box plot
                uncertainty_boxplot_path = os.path.join(frame_dir, 'uncertainty_boxplot.png')
                if not os.path.exists(uncertainty_boxplot_path):
                    action_data = calculate_action_uncertainty(all_results, trajectory, frame_index, low, high)
                    fig = create_uncertainty_boxplot(action_data)
                    fig.savefig(uncertainty_boxplot_path)
                    plt.close(fig)
        else:
            print(f"Skipping {trajectory} as it does not have language or results for both models.")
            

    
    ALREADY_PROCESSED = True
    print("All visualizations generated and stored.")
    
        

# Load data at app startup
ds_subset = load_dataset()
results = load_results()
mse_dict = compute_mse_dict(results, ds_subset)
generate_and_store_visualizations(results, ds_subset, 'val_pregenerated_visualizations')


# Set the title of the app
st.title('S1/S2 MSE Visualization and Comparison')

# List of completed trajectories
# trajectories_completed = list(results.keys())
trajectories_completed = list(mse_dict.keys())

# Dropdown to select the trajectory
selected_trajectory = st.selectbox('Select a trajectory', trajectories_completed)


    
if selected_trajectory:
    instruction = results[selected_trajectory]["OpenVLA"]["instruction"]
    st.markdown(f'## Task Instruction: {instruction}')
    st.write("  ")

    vla_mse = mse_dict[selected_trajectory]["VLA"]
    ecot_mse = mse_dict[selected_trajectory]["ECOT"]

    # Create Plotly figure
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(x=list(range(len(vla_mse))), y=vla_mse, name="VLA", line=dict(color="green")), secondary_y=False)
    fig.add_trace(go.Scatter(x=list(range(len(ecot_mse))), y=ecot_mse, name="ECOT", line=dict(color="red")), secondary_y=False)
    fig.update_layout(title_text=f"MSE over Time Steps for {selected_trajectory}", hovermode="x unified")
    fig.update_xaxes(title_text="Time Step", showgrid=True, gridwidth=1, gridcolor='LightGrey')
    fig.update_yaxes(title_text="Mean Squared Error", secondary_y=False, showgrid=True, gridwidth=1, gridcolor='LightGrey')

    st.plotly_chart(fig, use_container_width=True)

    # Slider to select frame
    frame_index = st.slider('Select Frame', 0, len(vla_mse)-1, 0)

    # Load and display images
    output_directory = 'val_pregenerated_visualizations'
    composite_image_path = os.path.join(output_directory, selected_trajectory, f'frame_{frame_index}', 'composite_image.png')
    mse_comparison_path = os.path.join(output_directory, selected_trajectory, f'frame_{frame_index}', 'mse_comparison.png')
    action_deltas_path = os.path.join(output_directory, selected_trajectory, f'frame_{frame_index}', 'action_deltas.png')
    frame_mse_comparison_path = os.path.join(output_directory, selected_trajectory, f'frame_{frame_index}', 'frame_mse_comparison.png')
    uncertainty_boxplot_path = os.path.join(output_directory, selected_trajectory, f'frame_{frame_index}', 'uncertainty_boxplot.png')


    st.image(composite_image_path, caption=f"Frame {frame_index}")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.image(mse_comparison_path)
    with col2:
        st.image(action_deltas_path)
    with col3:
        st.image(uncertainty_boxplot_path, caption="Uncertainty Across Action Dimensions")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.write("### ECoT Action")
        st.write(format_printable_action(results[selected_trajectory]["ECoT"]['actions'][frame_index]["action"]))
    with col2:
        st.image(frame_mse_comparison_path)
    with col3:
        st.write("### OpenVLA Action")
        st.write(format_printable_action(results[selected_trajectory]["OpenVLA"]['actions'][frame_index]["action"]))

    st.write("### Chain of Thought Response")
    formatted_reasoning_string = format_reasoning_string(results[selected_trajectory]["ECoT"]['actions'][frame_index]["generated_text"])
    st.write(formatted_reasoning_string)