"""
training logic using unlabeled target data
"""
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from model import GradientReversal
from utils import get_pairwise_sim_loss

def _get_accuracy(outputs,labels):
    "return accuracy in percentage"
    _, predicted = torch.max(outputs.data, 1)
    total = labels.size(0)
    correct = (predicted == labels.data).sum()
    return 100 * correct / total


def do_forward_pass(params, basic_feat_ext, code_gen, discriminator, imgs, labels):
    src_imgs, tgt_imgs = imgs
    labels_src,labels_tgt = labels
    basic_feat_src = basic_feat_ext(src_imgs)
    basic_feat_tgt = basic_feat_ext(tgt_imgs)

    # compute pairwise similarity loss
    code_feat_src = code_gen(basic_feat_src)
    pairwise_loss = get_pairwise_sim_loss(feats=code_feat_src,
                          labels=labels_src)

    # compute discriminator loss
    grad_reversal = GradientReversal.apply
    basic_feats = torch.cat([basic_feat_src,basic_feat_tgt])
    dis_outputs = discriminator(grad_reversal(basic_feats))
    ## calculate cross-entropy loss
    labels_for_calc_acc = [0 for _ in range(len(labels_src))] + [1 for _ in range(len(labels_tgt))]
    domain_labels = Variable(torch.LongTensor(np.array(labels_for_calc_acc,dtype=np.int64)),requires_grad=False)
    dis_loss = F.cross_entropy(input=dis_outputs, target=domain_labels)
    dis_accuracy = _get_accuracy(outputs=dis_outputs,
                                 labels=Variable(torch.LongTensor(np.array(labels_for_calc_acc,dtype=np.int64)),requires_grad=False))

    # add two loss together
    total_loss = torch.add(
        params.loss_coeff["code"] * pairwise_loss,
        params.loss_coeff["discriminator"] * dis_loss
    )

    return {
        "total_loss": total_loss,
        "code_loss": params.loss_coeff["code"] * pairwise_loss,
        "discriminator_loss": params.loss_coeff["discriminator"] * dis_loss,
        "discriminator_acc": dis_accuracy
    }