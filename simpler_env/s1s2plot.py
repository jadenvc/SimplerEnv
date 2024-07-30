
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt

def process_actions(path):
    """Load actions from a pickle file."""
    fp = os.path.join(path, "policy_out.pkl")
    with open(fp, "rb") as f:
        act_list = pickle.load(f)
    if isinstance(act_list[0], dict):
        act_list = [x["actions"] for x in act_list]
    return act_list

def read_arrays_from_file(filename):
    """Read action arrays from a text file."""
    arrays = []
    with open(filename, 'r') as f:
        lines = f.readlines()
        for line in lines:
            array = np.fromstring(line, sep=' ')
            arrays.append(array)
    return arrays

def calculate_mse(actions_gt, actions_model):
    """Calculate the mean squared error between ground truth and model actions."""
    mse_list = [np.mean((gt - model) ** 2) for gt, model in zip(actions_gt, actions_model)]
    return mse_list

# traj number
tj = 0
gt_path = "/iliad/group/datasets/bridgedata_v2/raw/bridge_data_v2/datacol1_toykitchen1/many_skills/0/2023-03-15_14-35-28/raw/traj_group0/traj" + str(tj)+"/"
gt_actions = process_actions(gt_path)

vla_actions = read_arrays_from_file("simpler_env/test_data/vla0.txt")
ecot_actions = read_arrays_from_file("simpler_env/test_data/ecot0.txt")

# Compute MSE
vla_mse = calculate_mse(gt_actions, vla_actions)
ecot_mse = calculate_mse(gt_actions, ecot_actions)


# Print MSE values
print(f"Average MSE between GT and VLA: {vla_mse}")
print(f"Average MSE between GT and ECOT: {ecot_mse}")
# Plotting MSE
plt.figure(figsize=(12, 6))
plt.plot(vla_mse, label='VLA', color='green')
plt.plot(ecot_mse, label='ECOT', color='red')
plt.xlabel('Time Step')
plt.ylabel('Mean Squared Error')
plt.title('MSE over Time Steps for VLA and ECOT Models')
plt.legend()
plt.grid(True)

# Create directory if it doesn't exist
os.makedirs('simpler_env/action_plots', exist_ok=True)

# Save the plot to action_plots
plt.savefig('simpler_env/action_plots/plot.png')
plt.show()