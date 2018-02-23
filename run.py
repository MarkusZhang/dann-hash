"""
Run domain adversarial net for hashing
"""

import os,pickle
import torch
from torch.autograd import Variable
import params
import model as test_model
from utils import LoggerGenerator, save_param_module
from train import training
from train_logic import do_forward_pass
from ml_test.testing import run_simple_test
from ml_toolkit.log_analyze import plot_trend_graph

# path to save models and logs
root_save_dir = "saved_models/first"
root_log_dir = "log"

# 1. set the common params
params.hash_size = 16
params.source_data_path = "F:/data/mnist/mini/training"
params.target_data_path = "F:/data/mnist_m/mini/train-10-percent"
params.test_data_path = {
    "query": "F:/data/mnist_m/mini/query-db-split/query",
    "db": "F:/data/mnist_m/mini/query-db-split/db"
}
params.batch_size = 30
params.iterations = 100
params.learning_rate = 0.001
params.use_dropout = False

save_param_module(params=params,save_to=os.path.join(root_save_dir, "common-params.txt"))

# choices
loss_coeff_choices = [
    {"code":0.5,"discriminator":0.5}
]

def train(params,id):
    "this is to run the training once and get the final loss"
    # make the folders
    save_model_path = os.path.join(root_save_dir, str(id))
    save_result_path = os.path.join(save_model_path,"test_results")
    save_log_path = os.path.join(root_log_dir, "{}.txt".format(id))
    if (not os.path.exists(save_model_path)): os.makedirs(save_model_path)
    if (not os.path.exists(save_result_path)): os.makedirs(save_result_path)

    # perform training
    train_results = training(params=params, logger=LoggerGenerator.get_logger(
        log_file_path=save_log_path), save_model_to=save_model_path,train_func=do_forward_pass)

    # plot loss vs. iterations
    lines = [str(l) for l in train_results["code_loss_records"]]
    plot_trend_graph(var_names=["code loss"], var_indexes=[-1], var_types=["float"], var_colors=["r"], lines=lines,
                     title="code loss",save_to=os.path.join(save_result_path,"train-code_loss.png"),show_fig=False)
    lines = [str(l) for l in train_results["discr_loss_records"]]
    plot_trend_graph(var_names=["discriminator loss"], var_indexes=[-1], var_types=["float"], var_colors=["r"], lines=lines,
                     title="discriminator loss", save_to=os.path.join(save_result_path, "train-discriminator_loss.png"), show_fig=False)
    lines = [str(l) for l in train_results["discr_acc_records"]]
    plot_trend_graph(var_names=["discriminator acc"], var_indexes=[-1], var_types=["float"], var_colors=["r"], lines=lines,
                     title="discriminator acc", save_to=os.path.join(save_result_path, "train-discriminator_acc.png"), show_fig=False)

    with open(os.path.join(save_result_path,"train_records.txt"),"w") as f:
        f.write(str(train_results))
    print("finish training for parameter set #{}".format(id))

    # perform testing
    results = run_simple_test(params=params, saved_model_path=save_model_path,model_def=test_model)
    # save test results
    results["records"]["precision-recall-curve.jpg"].save(os.path.join(save_result_path,"precision-recall.png"))
    with open(os.path.join(save_result_path,"metrics.txt"),"w") as f:
        f.write(str(results["results"]))

    print("finish testing for parameter set #{}".format(id))


def try_all():
    for i,loss_coeff in enumerate(loss_coeff_choices):
        params.loss_coeff = loss_coeff
        train(params=params,id=i+1)

if __name__ == "__main__":
    try_all()