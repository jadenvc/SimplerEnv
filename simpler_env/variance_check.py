import json
import numpy as np



# Call all 10 json data, and create different instaces of it to call from

# Load the json data

different_data = []
data_dir = 'eval_results/'

for i in range(10):
    file = data_dir + str(i) + '.json'
    with open(file) as f:
        data = json.load(f)
        different_data.append(data)


# Create a function that will calculate the variance of the data

