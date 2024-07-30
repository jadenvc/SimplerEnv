import numpy as np
import pickle
import os
import matplotlib.pyplot as plt

def plot_third_dimension(actions, index, label, color):
    third_dimension = [action[index] for action in actions if len(action) > 2]
    plt.plot(third_dimension, label=label, color=color)

def process_actions(path):  # gets actions
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

# traj number
tj = 0
gt_path = "/iliad/group/datasets/bridgedata_v2/raw/bridge_data_v2/datacol1_toykitchen1/many_skills/0/2023-03-15_14-35-28/raw/traj_group0/traj" + str(tj)+"/"
gt_actions = process_actions(gt_path)

vla_actions = read_arrays_from_file("simpler_env/test_data/vla0.txt")

ecot_actions = read_arrays_from_file("simpler_env/test_data/ecot0.txt")

# get l2 norm between vla and gt actions
vla_dif = []
ecot_dif = []
# breakpoint()
for i in range(len(gt_actions)):
    # vla = vla_actions[i]
    ecot = ecot_actions[i]
    gt = gt_actions[i]
    # vla_dif.append(np.linalg.norm(vla - gt))
    ecot_dif.append(np.linalg.norm(ecot - gt))


# print("average l2 norm between gt and vla: ", np.mean(vla_dif))
print("average l2 norm between gt and ecot: ", np.mean(ecot_dif))

# print()
# Plotting
for index in range (7):
    plt.figure(figsize=(10, 6))
    plot_third_dimension(gt_actions, index, 'Ground Truth', 'blue')
    plot_third_dimension(vla_actions, index, 'VLA', 'green')
    plot_third_dimension(ecot_actions, index, 'ECOT', 'red')

    plt.xlabel('Time Step')
    plt.ylabel(str(index) +' Dimension Value')
    # plt.title('Third Dimension of Actions Over Time')
    plt.legend()
    plt.grid(True)

    # Save the plot as a PNG file
    output_file = "simpler_env/action_plots/v"+str(tj) + "_"+str(index)+'_dimension_plot.png'
    plt.savefig(output_file)
    # clear the plot
    plt.clf()
# plt.show()