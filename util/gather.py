import torch
from torch import nn
import os

def gather_kd(labels, teacher_data, student_data, old_num_labels, new_num_labels):
    device = (torch.device('cuda')
              if student_data.is_cuda
              else torch.device('cpu'))
    Softmax = nn.Softmax(dim=-1)
    student_data = Softmax(student_data)
    student_old = student_data[labels < old_num_labels][:, :old_num_labels]
    teacher = teacher_data[labels < old_num_labels].to(device)
    student_new = student_data[labels >= old_num_labels]
    labels = labels[labels >= old_num_labels]

    return labels, student_new, student_old, teacher

def gather_rh(labels, teacher_data, student_data, old_num_labels):
    device = (torch.device('cuda')
              if student_data.is_cuda
              else torch.device('cpu'))
    student_old = student_data[labels < old_num_labels][:, :old_num_labels]
    teacher = teacher_data[labels < old_num_labels].to(device)
    student_new = student_data[labels >= old_num_labels]
    labels = labels[labels >= old_num_labels]
    return student_old, teacher


def gather_rh_ce(labels, teacher_data, student_data, old_num_labels):
    device = (torch.device('cuda')
              if student_data.is_cuda
              else torch.device('cpu'))

    student_old = student_data[labels < old_num_labels][:, :old_num_labels]
    teacher = teacher_data[labels < old_num_labels].to(device)
    student_new = student_data[labels >= old_num_labels]
    labels = labels[labels >= old_num_labels]
    return labels, student_new, student_old, teacher

def select_anchor_index(labels, mask):
    device = (torch.device('cuda')
              if mask.is_cuda
              else torch.device('cpu'))
    anchor = []
    for index, label in enumerate(labels):
        if label.item() != 0:
            anchor.append(index)
    # index of label != "O"
    anchor_index = torch.LongTensor(anchor).to(device)
    mask = torch.index_select(mask, 0, anchor_index)
    anchor = []
    for index, data in enumerate(mask):
        if data.sum() != 0:
            anchor.append(index)
    # index of label has positive samples
    anchor_index2 = torch.LongTensor(anchor).to(device)
    return anchor_index, anchor_index2

def select_label_token(labels, pad_token_label_id, mode):
    # device = (torch.device('cuda')
    #           if labels.is_cuda
    #           else torch.device('cpu'))
    labels_index = []
    idx = torch.arange(0, labels.shape[0])
    if mode == "memory_o":
        idx = idx[labels*(labels-pad_token_label_id) != 0]
        for index, label in enumerate(labels):
            if label.item() == 0 and label.item() != pad_token_label_id:
                labels_index.append(index)
    elif mode == "memory":
        # idx = idx[labels * (labels - pad_token_label_id) != 0]
        for index, label in enumerate(labels):
            if label.item() != 0 and label.item() != pad_token_label_id:
                labels_index.append(index)
    elif mode == "train":
        idx = idx[labels - pad_token_label_id != 0]
        for index, label in enumerate(labels):
            if label.item() != pad_token_label_id:
                labels_index.append(index)
    # index of label != "O"
    labels_index_list = torch.LongTensor(labels_index)

    return labels_index_list
