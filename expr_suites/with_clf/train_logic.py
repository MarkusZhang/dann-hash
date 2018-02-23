"""
Train using classifier loss
"""

"""
specific training procedures other than the general training framework
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
    labels_src = Variable(labels_src,requires_grad=False)
    labels_tgt = Variable(labels_tgt, requires_grad=False)
    basic_feat_src = basic_feat_ext(src_imgs)
    basic_feat_tgt = basic_feat_ext(tgt_imgs)

    # compute pairwise similarity loss
    code_feat_src, clf_out_src = code_gen(basic_feat_src)
    code_feat_tgt, clf_out_tgt = code_gen(basic_feat_tgt)
    clf_loss = F.cross_entropy(input=torch.cat([clf_out_src,clf_out_tgt]),
                               target=torch.cat([labels_src,labels_tgt]))
    clf_acc = _get_accuracy(outputs=torch.cat([clf_out_src,clf_out_tgt]),
                            labels=torch.cat([labels_src,labels_tgt]))

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
        params.loss_coeff["code"] * clf_loss,
        params.loss_coeff["discriminator"] * dis_loss
    )

    return {
        "total_loss": total_loss,
        "code_loss": params.loss_coeff["code"] * clf_loss,
        "clf_acc": clf_acc,
        "discriminator_loss": params.loss_coeff["discriminator"] * dis_loss,
        "discriminator_acc": dis_accuracy
    }