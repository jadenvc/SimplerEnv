import tensorflow as tf
import tensorflow_datasets as tfds



tf.random.set_seed(42)

# Load the dataset
ds_val = tfds.load('bridge_dataset', data_dir="/iliad/group/datasets/OXE_OCTO", split='val')

# Shuffle with a controlled buffer and take the subset
subset_size = 50
ds_subset = ds_val.shuffle(buffer_size=10000, seed=42, reshuffle_each_iteration=False).take(subset_size)


for inner_i, episode in enumerate(ds_subset):
    episode_metadata = episode["episode_metadata"]
    episode_name = f"traj{inner_i}_{episode_metadata['episode_id']}"
    
    print(f"traj{inner_i}_{episode_metadata['episode_id']}")
    
    print(episode_metadata['has_language'].numpy())
    
    image_list = []
    for step in episode["steps"]:
        image_list.append(step['observation']["image_0"].numpy())
        instruction = step['language_instruction'].numpy().decode('utf-8')
    print(instruction)
    