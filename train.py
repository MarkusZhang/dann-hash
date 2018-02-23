"""
This is the main training procedure
"""
import itertools
import os
import torch
from torch.autograd import Variable
from train_logic import do_forward_pass
from utils import get_data_loader, save_models, save_param_module
import model as sinno_model


def training(params, logger, train_func=do_forward_pass, save_model_to="",model_def=sinno_model):
    # create data loader
    source_loader = get_data_loader(data_path=params.source_data_path,params=params)
    target_loader = get_data_loader(data_path=params.target_data_path,params=params)

    # model components
    basic_feat_ext = model_def.BasicFeatExtractor(params=params)
    code_gen = model_def.CodeGen(params=params)
    discriminator = model_def.Discriminator(params=params)

    # optimizers
    learning_rate = params.learning_rate
    opt_basic_feat = torch.optim.Adam(basic_feat_ext.parameters(), lr=learning_rate)
    opt_code_gen = torch.optim.Adam(code_gen.parameters(), lr=learning_rate)
    opt_discriminator = torch.optim.Adam(discriminator.parameters(), lr=learning_rate)

    # for saving gradient
    grad_records = {}
    def save_grad(name):
        def hook(grad):
            grad_records[name] = grad
        return hook

    # training
    code_loss_records = []
    discr_loss_records = []
    discr_acc_records = []

    for i in range(params.iterations):
        # refresh data loader
        itertools.tee(source_loader)
        itertools.tee(target_loader)
        zipped_loader = enumerate(zip(source_loader, target_loader))
        acc_code_loss, acc_discr_loss, acc_discr_acc = 0,0,0
        logger.info("epoch {}/{} started".format(i,params.iterations))
        # train using minibatches
        for step, ((images_src, labels_src), (images_tgt, labels_tgt)) in zipped_loader:
            logger.info("batch {}".format(step))
            # clear gradients
            opt_basic_feat.zero_grad()
            opt_code_gen.zero_grad()
            opt_discriminator.zero_grad()

            # create input variables
            src_imgs = Variable(images_src).float()
            tgt_imgs = Variable(images_tgt).float()
            # src_lbls = Variable(labels_src)
            # tgt_lbls = Variable(labels_tgt)

            # call the specific training procedure to calculate loss
            train_out = train_func(params=params,basic_feat_ext=basic_feat_ext,code_gen=code_gen,discriminator=discriminator,
                       imgs=[src_imgs,tgt_imgs],labels=[labels_src,labels_tgt])
            total_loss,code_loss,discr_loss,discr_acc = train_out["total_loss"],train_out["code_loss"],train_out["discriminator_loss"],train_out["discriminator_acc"]

            # do weights update
            total_loss.backward()
            opt_basic_feat.step()
            opt_code_gen.step()
            opt_discriminator.step()

            acc_code_loss += code_loss.cpu().data.numpy()[0]
            acc_discr_loss += discr_loss.cpu().data.numpy()[0]
            acc_discr_acc += discr_acc

        code_loss_records.append(acc_code_loss/(step+1))
        discr_loss_records.append(acc_discr_loss/(step+1))
        discr_acc_records.append(acc_discr_acc/(step+1))
        logger.info("epoch {} | code loss: {}, discr loss: {}, discr acc: {}".format(i, acc_code_loss / (step+1),acc_discr_loss/(step+1), acc_discr_acc/(step+1)))


    # save model params
    save_models(models={"basic_feat_ext":basic_feat_ext,"code_gen":code_gen,"discriminator":discriminator},
                save_model_to=save_model_to,save_obj=True,save_params=True)
    # save the training settings
    save_param_module(params=params,save_to=os.path.join(save_model_to, "train_settings.txt"))
    logger.info("model saved")

    return {
        "code_loss_records": code_loss_records,
        "discr_loss_records": discr_loss_records,
        "discr_acc_records": discr_acc_records
    }

