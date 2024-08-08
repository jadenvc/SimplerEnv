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


ALREADY_PROCESSED = False
# Set wide mode
st.set_page_config(layout="wide")

results_file = 'forced_cot_results.json'

try:
    with open(results_file, 'r') as f:
        results = json.load(f)
except (json.JSONDecodeError, FileNotFoundError):
    raise Exception('"Error reading JSON file. Initializing an empty results dictionary."')

# Labels for the 7 numbers
action_element_labels = ['WV 1', 'WV 2', 'WV 3', 'RD 1', 'RD 2', 'RD 3', 'Gripper Closedness']
x_action_elements = np.arange(len(action_element_labels))  # the label locations
width = 0.35  # the width of the bars



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

def create_composite_image(image_path, gripper_position, visible_objects, reasoning_text):
    # Load the image
    image = Image.open(image_path)
    original_width, original_height = image.size

    # Calculate scaling factors (since we assume bounding boxes are for 256x256 images)
    scale_x = original_width / 256
    scale_y = original_height / 256

    # Create a new image with double the width to accommodate the text
    composite_image = Image.new('RGB', (2 * original_width, original_height), (255, 255, 255))
    composite_image.paste(image, (0, 0))

    # Draw on the image
    draw = ImageDraw.Draw(composite_image)

    # Draw gripper position (if available)
    if gripper_position:
        scaled_gripper_position = [int(gripper_position[0] * scale_x), int(gripper_position[1] * scale_y)]
        draw.ellipse(
            (scaled_gripper_position[0] - 5, scaled_gripper_position[1] - 5,
             scaled_gripper_position[0] + 5, scaled_gripper_position[1] + 5),
            fill='red', outline='red'
        )

    # Define font for text with increased size
    font1 = ImageFont.load_default().font_variant(size=22)  # Larger size for main text
    font2 = ImageFont.load_default().font_variant(size=16)
        
    # Draw visible objects bounding boxes and labels with semi-transparent fill
    for obj_label, bbox in visible_objects:
        scaled_bbox = [int(coord * scale_x if i % 2 == 0 else coord * scale_y) for i, coord in enumerate(bbox)]
        draw.rectangle(scaled_bbox, outline='blue', width=2)  # semi-transparent blue
        draw.text((scaled_bbox[0], scaled_bbox[1] - 20), obj_label, fill='blue', font=font2)

    # Draw reasoning text on the right side with text wrapping
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

def process_actions(path):
    fp = os.path.join(path, "policy_out.pkl")
    with open(fp, "rb") as f:
        act_list = pickle.load(f)
    if isinstance(act_list[0], dict):
        act_list = [x["actions"] for x in act_list]
    return act_list

def read_arrays_from_file(filename):
    arrays = []
    with open(filename, 'r') as f:
        lines = f.readlines()
        for line in lines:
            array = np.fromstring(line, sep=' ')
            arrays.append(array)
    return arrays

def calculate_mse(actions_gt, actions_model):
    mse_list = [np.mean((gt - model) ** 2) for gt, model in zip(actions_gt, actions_model)]
    return mse_list

def generate_and_store_visualizations(results, image_directory_base, output_directory):
    global ALREADY_PROCESSED
    if ALREADY_PROCESSED:
        return
    
    os.makedirs(output_directory, exist_ok=True)
    sorted_keys = sorted(list(results.keys()))
    
    for trajectory in tqdm(sorted_keys):
        trajectory_dir = os.path.join(output_directory, trajectory)
        os.makedirs(trajectory_dir, exist_ok=True)
        
        gt_actions = process_actions(f"{image_directory_base}{trajectory}/")
        vla_actions = [elem['raw_action'] for elem in results[trajectory]["OpenVLA"]['actions']]
        ecot_actions = [elem['raw_action'] for elem in results[trajectory]["ECoT"]['actions']]
        
        vla_mse = calculate_mse(gt_actions, vla_actions)
        ecot_mse = calculate_mse(gt_actions, ecot_actions)
        
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
        for frame_index in range(len(gt_actions)):
            frame_dir = os.path.join(trajectory_dir, f'frame_{frame_index}')
            os.makedirs(frame_dir, exist_ok=True)
            
            # Generate and save composite image
            composite_image_path = os.path.join(frame_dir, 'composite_image.png')
            if not os.path.exists(composite_image_path):
                frame_path = os.path.join(image_directory_base, f'{trajectory}/images0', f"im_{frame_index}.jpg")
                formatted_reasoning_string = format_reasoning_string(results[trajectory]["ECoT"]['actions'][frame_index]["generated_text"])
                gripper_position, visible_objects, cleaned_reasoning_string = parse_reasoning_string(formatted_reasoning_string)
                composite_image = create_composite_image(frame_path, gripper_position, visible_objects, cleaned_reasoning_string)
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
    
    ALREADY_PROCESSED = True
    print("All visualizations generated and stored.")
    
        
image_directory_base = "/iliad/group/datasets/bridgedata_v2/raw/bridge_data_v2/datacol1_toykitchen1/many_skills/0/2023-03-15_14-35-28/raw/traj_group0/"

output_directory = 'pregenerated_visualizations'
generate_and_store_visualizations(results, image_directory_base, output_directory)


# Set the title of the app
st.title('S1/S2 MSE Visualization and Comparison')

# Directories for plots and images
plot_directory = 'simpler_env/action_plots/first_pass/'

# List of completed trajectories
trajectories_completed = list(results.keys())

# Dropdown to select the trajectory
selected_trajectory = st.selectbox('Select a trajectory', trajectories_completed)

# Display the selected trajectory plot
if selected_trajectory:
    file_path = f"/iliad/group/datasets/bridgedata_v2/raw/bridge_data_v2/datacol1_toykitchen1/many_skills/0/2023-03-15_14-35-28/raw/traj_group0/{selected_trajectory}/lang.txt"
    with open(file_path, 'r') as file:
        instruction = file.readline().strip()
    
    st.markdown(f'## Task Instruction: {instruction}')
    st.write("  ")
    
    # For the MSE PLOT
    # Load actions and calculate MSE for each frame
    gt_actions = process_actions(f"{image_directory_base}{selected_trajectory}/")
    vla_actions = [elem['raw_action'] for elem in results[selected_trajectory]["OpenVLA"]['actions']]
    ecot_actions = [elem['raw_action'] for elem in results[selected_trajectory]["ECoT"]['actions']]

    # Calculate MSE
    vla_mse = calculate_mse(gt_actions, vla_actions)
    ecot_mse = calculate_mse(gt_actions, ecot_actions)

    # Create Plotly figure
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Add traces
    fig.add_trace(
        go.Scatter(x=list(range(len(vla_mse))), y=vla_mse, name="VLA", line=dict(color="green")),
        secondary_y=False,
    )

    fig.add_trace(
        go.Scatter(x=list(range(len(ecot_mse))), y=ecot_mse, name="ECOT", line=dict(color="red")),
        secondary_y=False,
    )

    # Set x-axis title
    fig.update_xaxes(title_text="Time Step")

    # Set y-axes titles
    fig.update_yaxes(title_text="Mean Squared Error", secondary_y=False)

    # Set title
    fig.update_layout(
        title_text=f"MSE over Time Steps for {selected_trajectory}",
        hovermode="x unified"
    )

    # Add grid
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightGrey')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGrey')

    # Display the plot
    st.plotly_chart(fig, use_container_width=True)
    
    # # Display MSE plot
    # mse_plot_path = os.path.join(output_directory, selected_trajectory, 'mse_plot.png')
    # st.image(mse_plot_path, caption=f"MSE Plot for {selected_trajectory}")

    # Load actions and calculate MSE for each frame
    gt_actions = process_actions(f"{image_directory_base}{selected_trajectory}/")
    
    vla_actions = [elem['raw_action'] for elem in results[selected_trajectory]["OpenVLA"]['actions']]
    ecot_actions = [elem['raw_action'] for elem in results[selected_trajectory]["ECoT"]['actions']]

    # Calculate MSE
    vla_mse = calculate_mse(gt_actions, vla_actions)
    ecot_mse = calculate_mse(gt_actions, ecot_actions)

    # Slider to select frame
    frame_index = st.slider('Select Frame', 0, len(gt_actions)-1, 0)

    formatted_reasoning_string = format_reasoning_string(results[selected_trajectory]["ECoT"]['actions'][frame_index]["generated_text"])


    frame_path = os.path.join(image_directory_base, f'{selected_trajectory}/images0', f"im_{frame_index}.jpg")
    
    # frame_image = Image.open(frame_path)
    
    # Parse reasoning string
    gripper_position, visible_objects, cleaned_reasoning_string = parse_reasoning_string(formatted_reasoning_string)

    # Create composite image
    # Display composite image
    composite_image_path = os.path.join(output_directory, selected_trajectory, f'frame_{frame_index}', 'composite_image.png')
    st.image(composite_image_path, caption=f"Frame {frame_index}")
    
    col1, col2, col3 = st.columns(3)

    with col1:
        mse_comparison_path = os.path.join(output_directory, selected_trajectory, f'frame_{frame_index}', 'mse_comparison.png')
        st.image(mse_comparison_path)

    with col2:
        action_deltas_path = os.path.join(output_directory, selected_trajectory, f'frame_{frame_index}', 'action_deltas.png')
        st.image(action_deltas_path)

    with col3:
        frame_mse_comparison_path = os.path.join(output_directory, selected_trajectory, f'frame_{frame_index}', 'frame_mse_comparison.png')
        st.image(frame_mse_comparison_path)

        
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("### ECoT Action")
        st.write(format_printable_action(results[selected_trajectory]["ECoT"]['actions'][frame_index]["action"]))
    with col2:
        st.write("### OpenVLA Action")
        st.write(format_printable_action(results[selected_trajectory]["OpenVLA"]['actions'][frame_index]["action"]))
    
        
    st.write("### Chain of Thought Response")
    st.write(formatted_reasoning_string)