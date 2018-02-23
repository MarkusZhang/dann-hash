"""
Utility functions for use in testing
"""
import numpy as np
import torch
import torchvision
import os
from ml_toolkit.data_process import batch_generator
from ml_toolkit.log_analyze import plot_trend_graph
import model as sinno_model
from torch import nn
from torch.autograd import Variable
from torchvision import transforms

def gen_hash_from_modules(models,imgs,params):
    "return the hash output"
    basic_feat_ext = models["basic_feat_ext"]
    code_gen = models["code_gen"]
    images = Variable(imgs).float()
    basic_feats = basic_feat_ext(images)
    code_gen_out = code_gen(basic_feats)
    if (hasattr(code_gen_out,"__iter__") and len(code_gen_out)==2):
        feats = code_gen_out[0]
    else:
        feats = code_gen_out
    return torch.sign(feats)


def _save_hash_code(item_set,fns,save_to,delimiter=","):
    "item_set should be list of {label:...,hash:...}"
    write_ls = []
    # write in the form of "filename\tlabel\thash"
    for i, fn in enumerate(fns):
        write_ls.append("{}{}{}{}{}".format(fn,delimiter,item_set[i]["label"],delimiter,item_set[i]["hash"]))
    open(save_to,"w").write("\n".join(write_ls))


def _get_data_loader(path, params, shuffle=False, use_batch=True):
    # image loading
    preprocess = transforms.Compose([
        transforms.Scale(params.image_scale),
        transforms.ToTensor(),
        transforms.Normalize(mean=params.dataset_mean, std=params.dataset_std)
    ])

    dataset = torchvision.datasets.ImageFolder(root=path, transform=preprocess)
    if (use_batch):
        batch_size = params.test_batch_size
    else:
        batch_size = len(dataset)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    img_filenames = [t[0] for t in dataset.imgs]
    return loader, img_filenames


def _construct_hash_function(models, params, binary=True):
    "return a function that takes in images and return hash code, if `binary` is False, will return real-value outputs delimited by comma"
    def hash_function(imgs):
        hash_outputs = gen_hash_from_modules(models=models,imgs=imgs,params=params)
        hash_str_ls = []
        for output in hash_outputs:
            if (binary):
                output = output.data.cpu().numpy().astype(np.int8)
                output[output == -1] = 0
                hash_str = "".join(output.astype(np.str))
            else:
                hash_str = ",".join(output.astype(np.str))
            hash_str_ls.append(hash_str)
        return hash_str_ls

    return hash_function


def _create_label_hash_dicts(hash_ls, label_ls):
    "return a list of dict {'label':...,'hash':'001010...'}"
    assert len(hash_ls) == len(label_ls)
    return [
        {"label":label_ls[i],"hash":hash_ls[i]}
        for i in range(len(hash_ls))
    ]


def _model_files_exist(path):
    files = ["basic_feat_ext.model","code_gen.model"]
    return all([os.path.exists(os.path.join(path,f)) for f in files])

def _load_models_from_path(saved_model_path, params, test_mode = True, model_def_module=sinno_model,use_model_file=True):
    "return a dict {'basic_feat_ext':model_obj, 'shared_feat_gen': .., 'specific_feat_gen':..}"
    if (_model_files_exist(path=saved_model_path) and use_model_file):
        print("Loading models using .model files")
        basic_ext = torch.load("{}/{}".format(saved_model_path,"basic_feat_ext.model"))
        code_gen = torch.load("{}/{}".format(saved_model_path, "code_gen.model"))
    # set models to eval mode if we are doing testing
    if (test_mode):
        basic_ext.eval()
        code_gen.eval()
        print("load models in eval mode")

    models = {
        "basic_feat_ext": basic_ext,
        "code_gen": code_gen,
    }
    return models