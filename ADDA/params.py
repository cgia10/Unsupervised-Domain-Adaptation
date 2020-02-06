# Adapted from https://github.com/corenel/pytorch-adda

"""Params for ADDA."""

# params for dataset and data loader
data_root = "./data/"
dataset_mean_value = 0.5
dataset_std_value = 0.5
dataset_mean = (dataset_mean_value, dataset_mean_value, dataset_mean_value)
dataset_std = (dataset_std_value, dataset_std_value, dataset_std_value)
batch_size = 200
image_size = 28

# params for source dataset
src_dataset = "SVHN"
src_encoder_restore = "./saved_models/"
src_classifier_restore = "./saved_models/"
src_model_trained = True

# params for target dataset
tgt_dataset = "MNIST"
tgt_encoder_restore = "./saved_models/"
tgt_model_trained = True

# params for setting up models
model_root = "./saved_models/"
d_input_dims = 500
d_hidden_dims = 500
d_output_dims = 2
d_model_restore = "./saved_models/"

# params for training network
num_gpu = 1
num_epochs_pre = 20
log_step_pre = 200
eval_step_pre = 2
save_step_pre = 2
num_epochs = 1
log_step = 200
save_step = 1
manual_seed = 1

# params for optimizing models
d_learning_rate = 2e-4
c_learning_rate = 2e-4
beta1 = 0.5
beta2 = 0.9
