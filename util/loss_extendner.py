from __future__ import print_function
import torch
from torch import nn
from torch.nn.modules import Module
from torch.nn import CrossEntropyLoss, KLDivLoss, NLLLoss
import sys
sys.path.append("/workspace/NERD")
from util.gather import select_anchor_index
import numpy as np

class ExtendNerLoss(Module):
    def __init__(self):
        super(ExtendNerLoss, self).__init__()

    def forward(self, student_new, labels, student_old, teacher, t):
        ce_loss_func = CrossEntropyLoss()
        kd_loss_func = KLDivLoss(reduction="batchmean")
        student_old = log_softmax_t(student_old, dim=-1, t=t)
        teacher = softmax_t(teacher, dim=-1, t=t)
        #kd_loss = kd_loss_func(teacher, student_old)
        kd_loss = kd_loss_func(student_old, teacher)
        if labels.shape[0] == 0:
            return kd_loss
        ce_loss = ce_loss_func(student_new, labels)
        loss = ce_loss + kd_loss
        return loss

class SupConLoss(Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07, topk_th=False):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature
        self.topk_th = topk_th

    def forward(self, features, labels=None, mask=None, entity_topk=None,
                ignore_index=CrossEntropyLoss(), per_types=6, aug_feature=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf
        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]  # n_views
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)  # [batch_size * n_views, feat_dim]
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature  # [batch_size * n_views, feat_dim]
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)   # [batch_size_new * n_views, batch_size * n_views]
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # select anchor
        mask[labels.view(-1) == ignore_index] = torch.zeros(mask.shape[1]).to(device)
        logits_mask[labels.view(-1) == ignore_index] = torch.zeros(logits_mask.shape[1]).to(device)
        if entity_topk is not None:
            # logits_mask[labels.view(-1) != 0, labels.view(-1) == 0] = torch.zeros(1).to(device)
            num_labels = torch.max(labels.view(-1)).item()+1
            old_num_labels = num_labels - per_types
            idx_o = torch.where(labels.view(-1) == 0)[0]
            logits_mask_o = torch.scatter(logits_mask, 1,
                                          idx_o.expand(logits_mask.shape[0], idx_o.shape[0]).long(),
                                          0)
            logits_mask_o = torch.where(labels >= old_num_labels, logits_mask_o, logits_mask)
            logits_mask_o = torch.scatter(logits_mask_o, 1,
                                          entity_topk.long(), 1)
            if aug_feature is not None:
                logits_mask_o[:, -aug_feature.shape[1]:] = \
                    torch.ones(logits_mask_o.shape[0], aug_feature.shape[1]).to(device)
            logits_mask = torch.where(labels >= old_num_labels, logits_mask_o, logits_mask)

        idx = torch.where(labels.view(-1)*mask.sum(dim=-1)*(labels.view(-1)+1) != 0)[0]
        # print(idx)
        logits_mask = logits_mask[idx]
        mask = mask[idx]
        logits = logits[idx]

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        if hasattr(torch.cuda, 'empty_cache'):
            torch.cuda.empty_cache()
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        if torch.isnan(mean_log_prob_pos).sum()>0:
            print("***mean_log_prob_pos is nan***")
        
        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, logits.shape[0]).mean()

        return loss

class SupConLoss_o(Module):
    # contrastive learning for "O"
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07, topk_th=False):
        super(SupConLoss_o, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature
        self.topk_th = topk_th
    def forward(self, features, labels=None, topk=None, negative_topk=None,entity_topk=None,
                mask=None, ignore_index=CrossEntropyLoss().ignore_index, per_types=6,
                aug_feature=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf
        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]  # n_views
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)  # [batch_size * n_views, feat_dim]
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature  # [batch_size * n_views, feat_dim]
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits

        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)  # [batch_size_new * n_views, batch_size * n_views]
        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # choose positive and negative token for type 'O'
        if negative_topk is None and topk is not None:
            topk = topk.view(-1, topk.shape[-1])  # (bsz*sql, topk_len)
            # labels == 'O': positive topk
            mask[labels.view(-1) == ignore_index] = torch.zeros(mask.shape[1]).to(device)
            if self.topk_th is False:
                mask_o = torch.scatter(torch.zeros_like(mask), 1, topk.long(), 1)
            else:
                mask_o = torch.zeros_like(mask) + topk
            mask = torch.where(labels == 0, mask_o, mask)
            logits_mask[labels.view(-1) == ignore_index] = torch.zeros(logits_mask.shape[1]).to(device)
            idx_no_o = torch.where(labels.view(-1) != 0)[0]
            if self.topk_th is False:
                no_negative = torch.cat((topk,
                                        idx_no_o.expand(topk.shape[0], idx_no_o.shape[0])), dim=1)
                logits_mask_o = torch.scatter(torch.zeros_like(logits_mask), 1,
                                            no_negative.long(), 1)
            else:
                no_negative = idx_no_o.expand(logits_mask.shape[0], idx_no_o.shape[0])
                logits_mask_o = torch.scatter(torch.zeros_like(logits_mask), 1,
                                              no_negative.long(), 1)
                logits_mask_o = logits_mask_o + topk
            logits_mask = torch.where(labels == 0, logits_mask_o, logits_mask)
            logits_mask = torch.scatter(logits_mask, 1,
                                        torch.arange(batch_size * anchor_count).view(-1, 1).to(device), 0)

        if negative_topk is not None and topk is not None:
            topk = topk.view(-1, topk.shape[-1])
            negative_topk = negative_topk.view(-1, negative_topk.shape[-1]) # (bsz*sql, negative_topk_len)
            mask[labels.view(-1) == ignore_index] = torch.zeros(mask.shape[1]).to(device)
            mask_o = torch.scatter(torch.zeros_like(mask), 1, topk.long(), 1)
            mask = torch.where(labels == 0, mask_o, mask)
            idx_no_o = torch.where(labels.view(-1) != 0)[0]
            negative_topk = torch.cat((topk, negative_topk,
                                       idx_no_o.expand(negative_topk.shape[0], idx_no_o.shape[0])), dim=1)
            logits_mask_o = torch.scatter(torch.zeros_like(logits_mask), 1, negative_topk.long(), 1)
            logits_mask = torch.where(labels == 0, logits_mask_o, logits_mask)
            logits_mask = torch.scatter(logits_mask, 1,
                                        torch.arange(batch_size * anchor_count).view(-1, 1).to(device), 0)

            # set different temperature for "O" negative top k tokens
            temp_o = torch.scatter(torch.ones_like(anchor_dot_contrast), 1,
                                   negative_topk.long(), (self.temperature+0.02)/self.temperature)
            temp = torch.where(labels == 0, temp_o, torch.ones_like(anchor_dot_contrast))
            anchor_dot_contrast = torch.div(anchor_dot_contrast, temp)
            # topk_idx = 0
            
        # select negative token for entity types
        if entity_topk is not None:
            num_labels = torch.max(labels.view(-1)).item() + 1
            old_num_labels = num_labels - per_types
            idx_o = torch.where(labels.view(-1) == 0)[0]
            logits_mask_o = torch.scatter(logits_mask, 1,
                                          idx_o.expand(logits_mask.shape[0], idx_o.shape[0]).long(),
                                          0)
            logits_mask_o = torch.where(labels >= old_num_labels, logits_mask_o, logits_mask)
            logits_mask_o = torch.scatter(logits_mask_o, 1,
                                          entity_topk.long(), 1)
            if aug_feature is not None:
                logits_mask_o[:, -aug_feature.shape[1]:] =\
                    torch.ones(logits_mask_o.shape[0], aug_feature.shape[1]).to(device)
            logits_mask = torch.where(labels >= old_num_labels, logits_mask_o, logits_mask)

        # delete the anchors that don not have positive data
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()
        mask[labels.view(-1) == ignore_index] = torch.zeros(mask.shape[1]).to(device)
        logits_mask[labels.view(-1) == ignore_index] = torch.zeros(logits_mask.shape[1]).to(device)
        idx = torch.where(mask.sum(dim=1)*(labels.view(-1)+1) != 0)[0]
        logits_mask = logits_mask[idx]
        mask = mask[idx]
        logits = logits[idx]

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        if hasattr(torch.cuda, 'empty_cache'):
            torch.cuda.empty_cache()
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        if torch.isnan(mean_log_prob_pos).sum()>0:
            print("***mean_log_prob_pos is nan***")
        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, len(idx)).mean()

        return loss


class KdLoss(Module):
    def __init__(self):
        super(KdLoss, self).__init__()

    def forward(self, student_features, teacher_features, t):
        loss_func = KLDivLoss(reduction="batchmean")
        student_old = log_softmax_t(student_features, dim=-1, t=t)
        teacher = softmax_t(teacher_features, dim=-1, t=t)
        loss = loss_func(student_old, teacher)
        return loss

class BceLoss(Module):
    def __init__(self, o_weight=None):
        super(BceLoss, self).__init__()
        self.o_weight = o_weight
    def forward(self, output, labels, num_labels, old_target=None, cal_O=False):
        pad_token_label_id = CrossEntropyLoss().ignore_index

        pos_weight = torch.ones(num_labels).to(output.device)
        if self.o_weight is not None:
            pos_weight[0] = self.o_weight
        if not cal_O:
            pos_weight = pos_weight[1:]
        criterion = torch.nn.BCEWithLogitsLoss(reduction='mean', pos_weight=pos_weight)
        idx = torch.where(labels >= 0)[0]
        valid_output = output[idx] # (b*l, num_labels)
        valid_labels = labels[idx]
        target = get_one_hot(valid_labels, num_labels) # (b*l,num_labels)

        if old_target is None:
            if not cal_O:
                target = target[:, 1:]
                valid_output = valid_output[:, 1:]
            return criterion(valid_output, target)

        else:
            old_target = old_target[idx]
            old_target = torch.sigmoid(old_target)
            old_task_size = old_target.shape[1]
            mask_O = (valid_labels < old_task_size).view(-1,1).repeat(1, num_labels) # (b*l,num_labels)
            mask_new = (valid_labels >= old_task_size).view(-1,1).repeat(1, num_labels)

            target_new = target.clone()
            target_new[:, :old_task_size] = 0
            target_new = target_new * mask_new

            target_O = target.clone()
            target_O[:, :old_task_size] = old_target
            target_O = target_O * mask_O

            target = target_new + target_O

            if not cal_O:
                target = target[:, 1:]
                valid_output = valid_output[:, 1:]

            return criterion(valid_output, target)

class BceLossNoKd(Module):
    def __init__(self, o_weight=None):
        super(BceLossNoKd, self).__init__()
        self.o_weight = o_weight

    def forward(self, output, labels, num_labels, cal_O=False):

        pos_weight = torch.ones(num_labels).to(output.device)
        if self.o_weight is not None:
            pos_weight[0] = self.o_weight
        if not cal_O:
            pos_weight = pos_weight[1:]
        criterion = torch.nn.BCEWithLogitsLoss(reduction='mean', pos_weight=pos_weight)

        idx = torch.where(labels >= 0)[0]
        valid_output = output[idx] # (b*l, num_labels)
        valid_labels = labels[idx]
        target = get_one_hot(valid_labels, num_labels) # (b*l,num_labels)

        if not cal_O:
            target = target[:, 1:]
            valid_output = valid_output[:, 1:]
        return criterion(valid_output, target)


def softmax_t(inputs, dim, t):
    inputs = inputs.div(t)
    Softmax = nn.Softmax(dim=dim)
    outputs = Softmax(inputs)
    return outputs


def log_softmax_t(inputs, dim, t):
    inputs = inputs.div(t)
    LogSoftmax = nn.LogSoftmax(dim=dim)
    outputs = LogSoftmax(inputs)
    return outputs

def get_one_hot(target, num_class):
    one_hot = torch.zeros(target.shape[0], num_class).to(target.device)
    one_hot = one_hot.scatter(dim=1, index=target.long().view(-1, 1), value=1.)

    return one_hot

from torch.nn import functional as F

def lwf_criterion(outputs, targets, outputs_old=None, T=2):
    """Returns the loss value"""
    idx = torch.where(targets >= 0)[0]
    valid_output = outputs[idx]  # (b*l, num_labels)
    targets = targets[idx]
    old_target = outputs_old

    if old_target is None:
        criterion = CrossEntropyLoss()
        return criterion(valid_output, targets)

    else:
        Softmax = nn.Softmax(dim=-1)
        valid_output = Softmax(valid_output)
        criterion = NLLLoss()
        old_target = old_target[idx]
        old_task_size = old_target.shape[1]

        loss_old = cross_entropy_t(valid_output[:, :old_task_size], old_target, exp=1.0 / T)

        loss_new = criterion(torch.log(valid_output), targets)

        return loss_old + loss_new

def cross_entropy_t(outputs, targets, exp=1.0, size_average=True, eps=1e-5):
    """Calculates cross-entropy with temperature scaling"""
    device = outputs.device
    out = F.softmax(outputs, dim=1).to(device)
    tar = F.softmax(targets, dim=1).to(device)
    if exp != 1:
        out = out.pow(exp)
        out = out / out.sum(1).view(-1, 1).expand_as(out)
        tar = tar.pow(exp)
        tar = tar / tar.sum(1).view(-1, 1).expand_as(tar)
    out = out + eps / out.size(1)
    out = (out / out.sum(1).view(-1, 1).expand_as(out)).to(device)
    ce = -(tar * out.log()).sum(1)
    if size_average:
        ce = ce.mean()
    return ce

def ce_bft_criterion(student_data, labels, teacher_data, old_num_labels, T=2):
    device = student_data.device
    Softmax = nn.Softmax(dim=-1)
    student_data = Softmax(student_data)
    # print("labels_shape:{}".format(labels.shape))
    teacher = teacher_data[labels >= old_num_labels][:,old_num_labels:].to(device)
    # teacher = teacher[:,old_num_labels:]
    student_kd = student_data[labels >= old_num_labels][:,old_num_labels:]
    # student_kd = student_kd[:, old_num_labels:]
    student_ce = student_data[labels > 0]
    labels_ce = labels[labels > 0]
    # print(teacher.shape)
    # print(student_kd.shape())
    nll_loss_fct = NLLLoss()
    kd_loss_fct = KdLoss()
    ce_loss = nll_loss_fct(torch.log(student_ce), labels_ce)
    kd_loss = kd_loss_fct(student_kd, teacher, T)
    return ce_loss+kd_loss
