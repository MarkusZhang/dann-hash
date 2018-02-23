########################################
# parameters for image preprocessing
########################################
dataset_mean_value = 0.5
dataset_std_value = 0.5
dataset_mean = (dataset_mean_value, dataset_mean_value, dataset_mean_value)
dataset_std = (dataset_std_value, dataset_std_value, dataset_std_value)

# image-scale: 100 for office, 28 for mnist
image_scale = 28
shuffle_batch = True
test_batch_size = 100
precision_radius = 2