# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Fine-tuning the library models for named entity recognition on CoNLL-2003 (Bert). """

from __future__ import absolute_import, division, print_function

import argparse
import glob
import logging
import os
import random
import json
import numpy as np
import torch
from seqeval.metrics import precision_score, recall_score, f1_score
# from tensorboardX import SummaryWriter
import sys
sys.path.append("/workspace/NERD")
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
from util.supervised_util import convert_examples_to_features, get_labels, get_labels_incr, read_examples_from_file, get_labels_dy

from transformers import AdamW, get_linear_schedule_with_warmup
from transformers import WEIGHTS_NAME, BertConfig, BertForTokenClassification, BertTokenizer
from datasets import load_metric
from util.metric import compute_metrics
from model.supcon_net import MySftBertModel
from util.ncm_classifier import NcmClassification, NNClassification
from util.gather import select_label_token
# os.environ["CUDA_VISIBLE_DEVICES"]="2"

logger = logging.getLogger(__name__)

# ALL_MODELS = sum(
#     (tuple(conf.pretrained_config_archive_map.keys()) for conf in (BertConfig, )),
#     ())

# bert_create is used for load previous model, which classifier's weight and bias are different with current.
# bert is default setting
MODEL_CLASSES = {
    "bert": (BertConfig, MySftBertModel, BertTokenizer)
}


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def train(args, train_dataset, train_dataloader, model, tokenizer, labels, pad_token_label_id, data_dir, output_dir, t_logits, out_new_labels):
    """ Train the model """
    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         "weight_decay": args.weight_decay},
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total)
    # scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=t_total)
    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank,
                                                          find_unused_parameters=True)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                args.train_batch_size * args.gradient_accumulation_steps * (
                    torch.distributed.get_world_size() if args.local_rank != -1 else 1))
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    train_iterator = trange(int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0])
    set_seed(args)  # Added here for reproductibility (even between python 2 and 3)
    #t_logits.to(args.device)
    num_labels = len(labels)
    top_emissions = None
    negative_top_emissions = None
    loss_name = args.loss_name1
    for epoch in train_iterator:
        if epoch >= args.start_train_o_epoch:
            prototype_dists = get_rehearsal_prototype(args, model, tokenizer, labels,
                                       pad_token_label_id, mode="topk",
                                       data_dir=data_dir)
        epoch_iterator = tqdm(train_dataloader, desc="Iterator", disable=args.local_rank not in [-1, 0])
        for step, batch in enumerate(epoch_iterator):
            model.train()
            batch = tuple(t.to(args.device) for t in batch)
            if num_labels-1 > args.per_types:
                t_logits_step = t_logits[step]
                #print(t_logits_step.size())
                t_logits_step.to(args.device)
                new_labels = out_new_labels[step * args.train_batch_size:step * args.train_batch_size + len(batch[3])]
                new_labels = torch.tensor(new_labels).to(args.device)
            else:
                t_logits_step = None
                new_labels = batch[3]
            if epoch >= args.start_train_o_epoch:
                loss_name = args.loss_name2
                cls = NNClassification()
                encodings, encoding_labels = get_token_features_and_labels(args, model, batch)
                top_emissions_step, _ = cls.get_top_emissions_with_th(encodings,
                                                                           encoding_labels,
                                                                           th_dists=torch.median(
                                                                               prototype_dists).item())

            else:
                top_emissions_step = top_emissions

            inputs = {"input_ids": batch[0],
                      "attention_mask": batch[1],
                      "token_type_ids": batch[2] if args.model_type in ["bert", "xlnet"] else None,
                      # XLM and RoBERTa don"t use segment_ids
                      "labels": new_labels,
                      "t_logits": t_logits_step,
                      "mode": "train",
                      "loss_name": loss_name,
                      "top_emissions": top_emissions_step,
                      "topk_th": True
                      }
            outputs = model(**inputs)
            loss = outputs[0]  # model outputs are always tuple in pytorch-transformers (see doc)

            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                # model.zero_grad()
                optimizer.zero_grad()
                global_step += 1

                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    # Log metrics
                    if args.local_rank == -1 and args.evaluate_during_training:  # Only evaluate when single GPU otherwise metrics may not average well
                        _, results, _ = evaluate(args, model, tokenizer, labels, pad_token_label_id, mode="dev",
                                                 data_dir=data_dir)
                    logging_loss = tr_loss

                if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
                    # Save model checkpoint
                    output_dir = os.path.join(output_dir, "checkpoint-{}".format(global_step))
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    model_to_save = model.module if hasattr(model, "module") else model  # Take care of distributed/parallel training
                    model_to_save.save_pretrained(output_dir)
                    torch.save(args, os.path.join(output_dir, "training_args.bin"))
                    logger.info("Saving model checkpoint to %s", output_dir)

            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break

    return global_step, tr_loss / global_step


def evaluate(args, model, tokenizer, labels, pad_token_label_id, mode, data_dir, prefix=""):
    eval_dataset = load_and_cache_examples(args, tokenizer, labels, pad_token_label_id, mode=mode, data_dir=data_dir)
    support_dataset = load_and_cache_examples(args, tokenizer, labels, pad_token_label_id, mode="memory",
                                              data_dir=data_dir)
    support_o_dataset = load_and_cache_examples(args, tokenizer, labels, pad_token_label_id, mode="memory_o",
                                                data_dir=data_dir)
    train_dataset = load_and_cache_examples(args, tokenizer, labels, pad_token_label_id, mode="train",
                                            data_dir=data_dir)
    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(eval_dataset) if args.local_rank == -1 else DistributedSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)
    support_sampler = SequentialSampler(support_dataset) \
        if args.local_rank == -1 else DistributedSampler(support_dataset)
    support_o_sampler = SequentialSampler(support_o_dataset) \
        if args.local_rank == -1 else DistributedSampler(support_o_dataset)
    train_sampler = SequentialSampler(train_dataset) \
        if args.local_rank == -1 else DistributedSampler(train_dataset)
    support_dataloader = DataLoader(support_dataset, sampler=support_sampler, batch_size=args.eval_batch_size)
    support_o_dataloader = DataLoader(support_o_dataset, sampler=support_o_sampler, batch_size=args.eval_batch_size)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.eval_batch_size)
    # Eval!
    logger.info("***** Running evaluation %s *****", prefix)
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    preds = None  # 预测值
    out_label_ids = None  # 真实标签
    model.eval()
    eval_iterator = tqdm(eval_dataloader, desc="Evaluating")
    if args.cls_name == "ncm_dot" or args.cls_name == "ncm_pow":
        support_encodings, support_labels = get_support_encodings_and_labels_total(
            args, model, support_dataloader, support_o_dataloader, train_dataloader, pad_token_label_id)
    else:
        support_encodings, support_labels = get_support_encodings_and_labels(
            args, model, support_dataloader, support_o_dataloader, pad_token_label_id)
    # print(support_encodings.size())
    exemplar_means = get_exemplar_means(args, support_encodings, support_labels)


    for step, batch in enumerate(eval_iterator):
        batch = tuple(t.to(args.device) for t in batch)
        encodings, encoding_labels = get_token_encodings_and_labels(args, model, batch)
        if mode == "rehearsal":
            cls = NNClassification()
            support_encodings = support_encodings[(support_labels < len(labels) - args.per_types) &
                                                    (support_labels > 0)]
            support_labels = support_labels[(support_labels < len(labels) - args.per_types) &
                                            (support_labels > 0)]
            nn_preds, nn_emissions = cls.nn_classifier_dot_score(
                encodings, support_encodings, support_labels)
            prototype_dists = None
        if args.cls_name == "ncm_dot":
            cls = NcmClassification()
            nn_preds = cls.ncm_classifier_dot(encodings, support_encodings, support_labels, exemplar_means)
        elif args.cls_name == "linear":
            nn_preds, encoding_labels = get_token_logits_and_labels(args, model, batch)
        
        if preds is None:
            preds = nn_preds.detach().cpu().numpy()
            out_label_ids = encoding_labels.detach().cpu().numpy()
            if mode == "rehearsal":
                emissions = nn_emissions.detach().cpu().numpy()

        else:
            preds = np.append(preds, nn_preds.detach().cpu().numpy(), axis=0)
            out_label_ids = np.append(out_label_ids, encoding_labels.detach().cpu().numpy(), axis=0)
            if mode == "rehearsal":
                emissions = np.append(emissions, nn_emissions.detach().cpu().numpy(), axis=0)
        # memory management
        del nn_preds
        torch.cuda.empty_cache()
    # eval_loss = eval_loss / nb_eval_steps
    if mode == "rehearsal":
        return preds, emissions, out_label_ids, prototype_dists
    if args.cls_name == "linear":
        preds = np.argmax(preds, axis=2)
    label_map = {i: "I-"+label for i, label in enumerate(labels)}
    label_map[0] = "O"
    # print(label_map)
    out_label_list = [[] for _ in range(out_label_ids.shape[0])]
    preds_list = [[] for _ in range(out_label_ids.shape[0])]

    for i in range(out_label_ids.shape[0]):
        for j in range(out_label_ids.shape[1]):
            if out_label_ids[i, j] != pad_token_label_id:
                out_label_list[i].append(label_map[out_label_ids[i][j]])
                preds_list[i].append(label_map[preds[i][j]])
    metric = load_metric("seqeval")
    metric.add_batch(
        predictions=preds_list,
        references=out_label_list,
    )

    macro_results, micro_results, _ = compute_metrics(metric)

    logger.info("***** Eval macro results %s *****", prefix)
    for key in sorted(macro_results.keys()):
        logger.info("  %s = %s", key, str(macro_results[key]))

    logger.info("***** Eval micro results %s *****", prefix)
    for key in sorted(micro_results.keys()):
        logger.info("  %s = %s", key, str(micro_results[key]))

    return macro_results, micro_results, preds_list

def get_rehearsal_prototype(args, model, tokenizer, labels, pad_token_label_id, mode, data_dir, prefix=""):
    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    support_dataset = load_and_cache_examples(args, tokenizer, labels, pad_token_label_id, mode="memory",
                                              data_dir=data_dir)
    support_o_dataset = load_and_cache_examples(args, tokenizer, labels, pad_token_label_id, mode="memory_o",
                                                data_dir=data_dir)
    support_sampler = SequentialSampler(support_dataset) \
        if args.local_rank == -1 else DistributedSampler(support_dataset)
    support_o_sampler = SequentialSampler(support_o_dataset) \
        if args.local_rank == -1 else DistributedSampler(support_o_dataset)
    support_dataloader = DataLoader(support_dataset, sampler=support_sampler, batch_size=args.eval_batch_size)
    support_o_dataloader = DataLoader(support_o_dataset, sampler=support_o_sampler, batch_size=args.eval_batch_size)
    support_encodings, support_labels = get_support_features_and_labels(
        args, model, support_dataloader, support_o_dataloader, pad_token_label_id)

    prototype_dists = []

    from torch.nn import functional as F
    support_encodings = F.normalize(support_encodings)
    if mode == "topk":
        start_class = 1
        num_class = len(labels)
        th_para = 1
    else:
        start_class = 0
        num_class = len(labels) - args.per_types
        th_para = args.relabel_th
        th_reduction = args.relabels_th_reduction
    current_task_id = (len(labels) - 1) // args.per_types
    for i in range(start_class, num_class):
        support_reps_dists = torch.matmul(support_encodings[support_labels == i],
                                          support_encodings[support_labels == i].T)
        support_reps_dists = torch.scatter(support_reps_dists, 1,
                                           torch.arange(support_reps_dists.shape[0]).view(-1, 1).to(args.device),
                                           0.)
        if args.change_th and mode != "topk":
            task_id = (i-1) // args.per_types
            task_para = th_para - (current_task_id - task_id - 1) * th_reduction
        else:
            task_para = th_para
        prototype_dists.append(support_reps_dists[support_reps_dists > 0].view(-1).mean(-1)*task_para)
    # print(prototype_dists)
    prototype_dists = torch.stack(prototype_dists).to(args.device)
    return prototype_dists

def teacher_evaluate(args, train_dataloader, model, tokenizer, labels, pad_token_label_id, mode, data_dir, prefix=""):
    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    if mode == "train":
        eval_dataloader = train_dataloader
    elif mode == "dev":
        eval_dataset = load_and_cache_examples(args, tokenizer, labels, pad_token_label_id, mode=mode,
                                               data_dir=data_dir)

        # Note that DistributedSampler samples randomly
        eval_sampler = SequentialSampler(eval_dataset) if args.local_rank == -1 else DistributedSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

    # Eval!
    logger.info("***** Running teacher model evaluation %s *****", prefix)
    # logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    logits_list = []  #
    model.eval()
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        batch = tuple(t.to(args.device) for t in batch)
        logits, out_labels = get_token_logits_and_labels(args, model, batch)
            # eval_loss += tmp_eval_loss.item()
        nb_eval_steps += 1
        logits_list.append(logits.detach().cpu())
        torch.cuda.empty_cache()
    preds, emissions, out_label_ids, _ = evaluate(args, model, tokenizer, labels, pad_token_label_id, mode="rehearsal",
                             data_dir=data_dir)
    prototype_dists = get_rehearsal_prototype(args, model, tokenizer, labels,
                                              pad_token_label_id, mode="rehearsal",
                                              data_dir=data_dir)
    out_label_new_list = [[] for _ in range(out_label_ids.shape[0])]
    for i in range(out_label_ids.shape[0]):
        for j in range(out_label_ids.shape[1]):
            idx = preds[i][j]
            if emissions[i][j] > prototype_dists[idx].item() and \
                out_label_ids[i][j] < len(labels) - args.per_types:
                out_label_new_list[i].append(preds[i][j])
            else:
                out_label_new_list[i].append(out_label_ids[i][j])
    return logits_list, out_label_new_list

def load_and_cache_examples(args, tokenizer, labels, pad_token_label_id, mode, data_dir):
    if args.local_rank not in [-1, 0] and not evaluate:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    # Load data features from cache or dataset file
    cached_features_file = os.path.join(data_dir, "cached_{}_{}_{}".format(mode,
        list(filter(None, args.model_name_or_path.split("/"))).pop(),
        str(args.max_seq_length)))
    if os.path.exists(cached_features_file) and not args.overwrite_cache:
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
    else:
        logger.info("Creating features from dataset file at %s", data_dir)
        examples = read_examples_from_file(data_dir, mode)
        features = convert_examples_to_features(examples, labels, args.max_seq_length, tokenizer,
                                                cls_token_at_end=bool(args.model_type in ["xlnet"]),
                                                # xlnet has a cls token at the end
                                                cls_token=tokenizer.cls_token,
                                                cls_token_segment_id=2 if args.model_type in ["xlnet"] else 0,
                                                sep_token=tokenizer.sep_token,
                                                sep_token_extra=bool(args.model_type in ["roberta"]),
                                                # roberta uses an extra separator b/w pairs of sentences, cf. github.com/pytorch/fairseq/commit/1684e166e3da03f5b600dbb7855cb98ddfcd0805
                                                pad_on_left=bool(args.model_type in ["xlnet"]),
                                                # pad on the left for xlnet
                                                pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
                                                pad_token_segment_id=4 if args.model_type in ["xlnet"] else 0,
                                                pad_token_label_id=pad_token_label_id
                                                )
        if args.local_rank in [-1, 0]:
            logger.info("Saving features into cached file %s", cached_features_file)
            torch.save(features, cached_features_file)

    if args.local_rank == 0 and not evaluate:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_ids for f in features], dtype=torch.long)

    dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
    return dataset
def get_token_features_and_labels(args, model, batch):
    """
    Get token encoding using pretrained BERT-NER model as well as groundtruth label
    """
    batch = tuple(t.to(args.device) for t in batch)
    label_batch = batch[3]
    with torch.no_grad():
        inputs = {"input_ids": batch[0], "attention_mask": batch[1], "output_hidden_states": True,
                  "mode": "dev"}
        if model.config.model_type != "distilbert":
            inputs["token_type_ids"] = (
                batch[2] if model.config.model_type in ["bert", "xlnet"] else None
            )  # XLM and RoBERTa don"t use token_type_ids
        outputs = model(**inputs)
        features = outputs[2]  # last layer representations
    return features.view(label_batch.shape[0],label_batch.shape[1], -1), label_batch
def get_token_encodings_and_labels(args, model, batch):
    """
    Get token encoding using pretrained BERT-NER model as well as groundtruth label
    """
    batch = tuple(t.to(args.device) for t in batch)
    label_batch = batch[3]
    with torch.no_grad():
        inputs = {"input_ids": batch[0], "attention_mask": batch[1], "output_hidden_states": True,
                  "mode": "dev"}
        if model.config.model_type != "distilbert":
            inputs["token_type_ids"] = (
                batch[2] if model.config.model_type in ["bert", "xlnet"] else None
            )  # XLM and RoBERTa don"t use token_type_ids
        outputs = model(**inputs)
        hidden_states = outputs[1]  # last layer representations
    return hidden_states, label_batch

def get_token_logits_and_labels(args, model, batch):
    """
    Get token encoding using pretrained BERT-NER model as well as groundtruth label
    """
    batch = tuple(t.to(args.device) for t in batch)
    label_batch = batch[3]
    with torch.no_grad():
        inputs = {"input_ids": batch[0], "attention_mask": batch[1], "output_hidden_states": True,
                  "mode": "dev"}
        if model.config.model_type != "distilbert":
            inputs["token_type_ids"] = (
                batch[2] if model.config.model_type in ["bert", "xlnet"] else None
            )  # XLM and RoBERTa don"t use token_type_ids
        outputs = model(**inputs)
        logits = outputs[-1]  # last layer representations
    return logits, label_batch

def get_support_encodings_and_labels(args, model, support_loader, support_o_loader, pad_token_label_id):
    """
    Get token encodings and labels for all tokens in the support set
    """
    support_encodings, support_labels = [], []
    support_iterator = tqdm(support_loader, desc="Support data representations")
    for index, batch in enumerate(support_iterator):
        encodings, labels = get_token_encodings_and_labels(args, model, batch)
        encodings = encodings.view(-1, encodings.shape[-1])
        labels = labels.flatten()
        # filter out PAD tokens
        idx = torch.where(labels*(labels - pad_token_label_id) != 0)[0]
        support_encodings.append(encodings[idx])
        support_labels.append(labels[idx])
    support_o_iterator = tqdm(support_o_loader, desc="Support data representations")
    for _, batch in enumerate(support_o_iterator):
        encodings, labels = get_token_encodings_and_labels(args, model, batch)
        encodings = encodings.view(-1, encodings.shape[-1])
        labels = labels.flatten()
        # filter out PAD tokens
        idx = torch.where((labels-pad_token_label_id) != 0)[0]
        labels = labels[idx]
        encodings = encodings[idx]
        support_encodings.append(encodings[labels == 0])
        support_labels.append(labels[labels == 0])
        
    return torch.cat(support_encodings), torch.cat(support_labels)

def get_support_features_and_labels(args, model, support_loader, support_o_loader, pad_token_label_id):
    """
    Get token encodings and labels for all tokens in the support set
    """
    support_encodings, support_labels = [], []
    support_iterator = tqdm(support_loader, desc="Support data representations")
    for index, batch in enumerate(support_iterator):
        encodings, labels = get_token_features_and_labels(args, model, batch)
        encodings = encodings.view(-1, encodings.shape[-1])
        labels = labels.flatten()
        # filter out PAD tokens
        idx = torch.where(labels*(labels - pad_token_label_id) != 0)[0]
        support_encodings.append(encodings[idx])
        support_labels.append(labels[idx])
    support_o_iterator = tqdm(support_o_loader, desc="Support data representations")
    for _, batch in enumerate(support_o_iterator):
        encodings, labels = get_token_features_and_labels(args, model, batch)
        encodings = encodings.view(-1, encodings.shape[-1])
        labels = labels.flatten()
        # filter out PAD tokens
        idx = torch.where((labels-pad_token_label_id) != 0)[0]
        labels = labels[idx]
        encodings = encodings[idx]
        support_encodings.append(encodings[labels == 0])
        support_labels.append(labels[labels == 0])
        
    return torch.cat(support_encodings), torch.cat(support_labels)

def get_support_encodings_and_labels_total(args, model, support_loader, support_o_loader, train_loader, pad_token_label_id):
    """
    Get token encodings and labels for all tokens in the support set
    """
    support_encodings, support_labels = [], []
    train_iterator = tqdm(train_loader, desc="Support data representations")
    for index, batch in enumerate(train_iterator):
        encodings, labels = get_token_encodings_and_labels(args, model, batch)
        encodings = encodings.view(-1, encodings.shape[-1])
        labels = labels.flatten()
        # filter out PAD tokens
        idx = torch.where((labels - pad_token_label_id) != 0)[0]
        support_encodings.append(encodings[idx])
        support_labels.append(labels[idx])

    support_iterator = tqdm(support_loader, desc="Support data representations")
    for index, batch in enumerate(support_iterator):
        encodings, labels = get_token_encodings_and_labels(args, model, batch)
        encodings = encodings.view(-1, encodings.shape[-1])
        labels = labels.flatten()
        idx = torch.where(labels * (labels - pad_token_label_id) != 0)[0]
        support_encodings.append(encodings[idx])
        support_labels.append(labels[idx])
    support_o_iterator = tqdm(support_o_loader, desc="Support data representations")
    for _, batch in enumerate(support_o_iterator):
        encodings, labels = get_token_encodings_and_labels(args, model, batch)
        encodings = encodings.view(-1, encodings.shape[-1])
        labels = labels.flatten()
        # filter out PAD tokens
        idx = torch.where((labels - pad_token_label_id) != 0)[0]
        labels = labels[idx]
        encodings = encodings[idx]
        support_encodings.append(encodings[labels == 0])
        support_labels.append(labels[labels == 0])
    return torch.cat(support_encodings), torch.cat(support_labels)

def get_exemplar_means(args, support_reps, support_labels):
    exemplar_means = {}
    n_tags = torch.max(support_labels) + 1
    cls_exemplar = {cls: [] for cls in range(n_tags)}
    for x, y in zip(support_reps, support_labels):
        cls_exemplar[y.item()].append(x)
    for cls, exemplar in cls_exemplar.items():
        features = []
        # Extract feature for each exemplar in p_y
        for feature in exemplar:
            feature.data = feature.data / feature.data.norm()  # Normalize
            features.append(feature)
        if len(features) == 0:
            mu_y = torch.normal(0, 1, size=tuple(x.size())).to(args.device)
            mu_y = mu_y.squeeze()
        else:
            features = torch.stack(features)
            mu_y = features.mean(0).squeeze()
        mu_y.data = mu_y.data / mu_y.data.norm()  # Normalize
        exemplar_means[cls] = mu_y
    return exemplar_means

def train_and_eval(args, labels, num_labels, pad_token_label_id, model_name_or_path, output_dir, data_dir, step_id):
    # Load pretrained model and tokenizer

    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    logger.info("Training/evaluation parameters %s", args)
    # obtain features from teacher model

    # Training
    args.model_type = args.model_type.lower()
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(args.config_name if args.config_name else model_name_or_path,
                                          num_labels=num_labels)
    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name if args.tokenizer_name else model_name_or_path,
                                                do_lower_case=args.do_lower_case)
    model = model_class.from_pretrained(model_name_or_path, from_tf=bool(".ckpt" in model_name_or_path),
                                        config=config)

    model.to(args.device)

    if args.do_train:
        train_dataset = load_and_cache_examples(args, tokenizer, labels, pad_token_label_id, mode="rehearsal",
                                                data_dir=data_dir)
        args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
        train_sampler = SequentialSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
        train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

        if step_id > 0:
            # get teacher model's features
            t_logits, out_new_labels = teacher_evaluate(args, train_dataloader, model, tokenizer, labels, pad_token_label_id,
                                          mode="train", data_dir=data_dir)
            model.new_classifier()
            model.to(args.device)
        else:
            t_logits = None
            out_new_labels = None

        global_step, tr_loss = train(args, train_dataset, train_dataloader, model, tokenizer, labels,
                                     pad_token_label_id, data_dir=data_dir, output_dir=output_dir,
                                     t_logits=t_logits, out_new_labels=out_new_labels)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)


    # Saving best-practices: if you use defaults names for the model, you can reload it using from_pretrained()
    if args.do_train and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        # Create output directory if needed
        if not os.path.exists(output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(output_dir)

        logger.info("Saving model checkpoint to %s", output_dir)
        # Save a trained model, configuration and tokenizer using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        model_to_save = model.module if hasattr(model,
                                                "module") else model  # Take care of distributed/parallel training
        model_to_save.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(args, os.path.join(output_dir, "training_args.bin"))

    # Evaluation
    # config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    results = {}
    if args.do_eval and args.local_rank in [-1, 0]:

        tokenizer = tokenizer_class.from_pretrained(output_dir, do_lower_case=args.do_lower_case)
        checkpoints = [output_dir]
        if args.eval_all_checkpoints:
            checkpoints = list(
                os.path.dirname(c) for c in sorted(glob.glob(output_dir + "/**/" + WEIGHTS_NAME, recursive=True)))
            logging.getLogger("pytorch_transformers.modeling_utils").setLevel(logging.WARN)  # Reduce logging
        logger.info("Evaluate the following checkpoints: %s", checkpoints)
        # exemplar_means_path = os.path.join(args.data_dir, "exemplar_means.json")

        for checkpoint in checkpoints:
            global_step = checkpoint.split("-")[-1] if len(checkpoints) > 1 else ""
            model = model_class.from_pretrained(checkpoint, mode="dev")
            model.to(args.device)
            train_dataloader=None
            result, _, _ = evaluate(args, model, tokenizer, labels, pad_token_label_id, mode="dev",
                                    data_dir=data_dir, prefix=global_step)
            if global_step:
                result = {"{}_{}".format(global_step, k): v for k, v in result.items()}
            results.update(result)
        output_eval_file = os.path.join(output_dir, "eval_results.txt")
        with open(output_eval_file, "w") as writer:
            for key in sorted(results.keys()):
                writer.write("{} = {}\n".format(key, str(results[key])))

    if args.do_predict and args.local_rank in [-1, 0]:
        tokenizer = tokenizer_class.from_pretrained(output_dir, do_lower_case=args.do_lower_case)
        model = model_class.from_pretrained(output_dir, mode="test")
        model.to(args.device)

        # data_dir = args.data_dir
        macro_results, micro_results, predictions = evaluate(args, model, tokenizer, labels,
                                                                          pad_token_label_id,
                                                                          mode="test", data_dir=data_dir)

        # Save results
        output_test_results_file = args.log_dir

        with open(output_test_results_file, "a") as writer:
            writer.write("{}\n".format(step_id))
            for key in sorted(macro_results.keys()):
                writer.write("macro_{} = {}\n".format(key, str(macro_results[key])))
            writer.write("\n")
            for key in sorted(micro_results.keys()):
                writer.write("micro_{} = {}\n".format(key, str(micro_results[key])))

        # Save predictions

        output_test_predictions_file = os.path.join(output_dir, "test_pred_gold.txt")
        
        with open(output_test_predictions_file, "w") as writer:
            with open(os.path.join(data_dir, "test.txt"), "r") as f:
                example_id = 0
                for line in f:
                    if line.startswith("-DOCSTART-") or line == "" or line == "\n":
                        writer.write(line)
                        if not predictions[example_id]:
                            example_id += 1
                    elif predictions[example_id]:
                        output_line = line.split()[0] + " " + predictions[example_id].pop(0)[2:] \
                                      + " " + line.split()[-1] + "\n"
                        writer.write(output_line)
                    else:
                        logger.warning("Maximum sequence length exceeded: No prediction for '%s'.", line.split()[0])

        return results

def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--data_dir", default=None, type=str, required=True,
                        help="The input data dir. Should contain the training files for the CoNLL-2003 NER task.")
    parser.add_argument("--model_type", default=None, type=str, required=True,
                        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()))
    parser.add_argument("--model_type_create", default=None, type=str, required=True,
                        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()))
    parser.add_argument("--model_type_eval", default=None, type=str, required=True,
                        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()))
    parser.add_argument("--model_name_or_path", default=None, type=str, required=True,
                        help="Path to pre-trained model or shortcut name")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")

    ## Other parameters
    parser.add_argument("--labels", default="", type=str,
                        help="Path to a file containing all labels. If not specified, CoNLL-2003 labels are used.")
    parser.add_argument("--config_name", default="", type=str,
                        help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name")
    parser.add_argument("--cache_dir", default="", type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")
    parser.add_argument("--max_seq_length", default=128, type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    
    ## Traning or evaluating parameters
    parser.add_argument("--do_train", action="store_true",
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action="store_true",
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_predict", action="store_true",
                        help="Whether to run predictions on the test set.")
    parser.add_argument("--change_th", action="store_true")
    parser.add_argument("--evaluate_during_training", action="store_true",
                        help="Whether to run evaluation during training at each logging step.")
    parser.add_argument("--memory_update", action="store_true",
                        help="Whether to update memory data.")
    parser.add_argument("--do_lower_case", action="store_true",
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--per_gpu_train_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    
    ## Epoch
    parser.add_argument("--num_train_epochs", default=3.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--start_train_o_epoch", default=3.0, type=float,
                        help="The number of training type 'O' epoch to perform.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument("--scale", type=int, default=25)
    parser.add_argument("--logging_steps", type=int, default=50,
                        help="Log every X updates steps.")
    parser.add_argument("--save_steps", type=int, default=50,
                        help="Save checkpoint every X updates steps.")
    parser.add_argument("--eval_all_checkpoints", action="store_true",
                        help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number")
    parser.add_argument("--no_cuda", action="store_true",
                        help="Avoid using CUDA when available")
    parser.add_argument("--overwrite_output_dir", action="store_true",
                        help="Overwrite the content of the output directory")
    parser.add_argument("--overwrite_cache", action="store_true",
                        help="Overwrite the cached training and evaluation sets")
    parser.add_argument("--seed", type=int, default=42,
                        help="random seed for initialization")
    
    ## Relabling parameters
    parser.add_argument("--relabel_th", type=float, default=1.0,required=True)
    parser.add_argument("--relabels_th_reduction", type=float, default=0.05, required=True)
    parser.add_argument("--fp16", action="store_true",
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument("--fp16_opt_level", type=str, default="O1",
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")
    parser.add_argument("--server_ip", type=str, default="", help="For distant debugging.")
    parser.add_argument("--server_port", type=str, default="", help="For distant debugging.")
    parser.add_argument("--loss_name1", type=str, default="", help="Name of entity-oriented loss function.")
    parser.add_argument("--loss_name2", type=str, default="", help="Name of entity-aware with 'O' loss function.")
    parser.add_argument("--cls_name", type=str, default="", help="Name of classifier.")
    ## Task parameters
    parser.add_argument("--nb_tasks", type=int, default=1, help="The number of tasks.")
    parser.add_argument("--start_step", type=int, default=0, help="The index of start step.")
    parser.add_argument("--log_dir", type=str, default="",
                        help="The logging directory where the test results will be written.")
    parser.add_argument("--per_types", type=int, default=0,
                        help="The number of each task.")
    parser.add_argument("--feat_dim", type=int, default=128,
                        help="The dimension of features.")
    parser.add_argument("--train_temp", type=int, default=2,
                        help="The distilling temperature in training parse.")
    parser.add_argument("--eval_temp", type=int, default=1,
                        help="The distilling temperature in inference parse.")
    args = parser.parse_args()

    if os.path.exists(args.output_dir) and os.listdir(
            args.output_dir) and args.do_train and not args.overwrite_output_dir:
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                args.output_dir))

    # Setup distant debugging if needed
    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd
        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 1
    args.device = device

    # Setup logging
    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
                        datefmt="%m/%d/%Y %H:%M:%S",
                        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
                   args.local_rank, device, args.n_gpu, bool(args.local_rank != -1), args.fp16)

    # Set seed
    set_seed(args)

    per_types = args.per_types
    # nb_tasks = 11

    # incremental learning setting
    all_metrics = torch.zeros([args.nb_tasks, args.nb_tasks, 3])
    average_f1 = torch.zeros([args.nb_tasks])
    memory_data = None
    output_test_results_file = args.log_dir
    with open(output_test_results_file, "a") as writer:
        writer.write("num_train_epochs={} start_train_o_epoch={} positive_top_k={} negative_top_k={}\n"
                     .format(args.num_train_epochs, args.start_train_o_epoch, args.top_k, args.negative_top_k))
        writer.write("confidence={}\n".format(args.confidence))
    for step_id in range(args.start_step, args.nb_tasks):
        labels = get_labels_dy(args.labels, per_types, step_id=step_id)
        num_labels = len(labels)
        # Use cross entropy ignore index as padding label id so that only real label ids contribute to the loss later
        pad_token_label_id = CrossEntropyLoss().ignore_index
        if step_id == 0:
            model_name_or_path = "bert-base-uncased"
        else:
            model_name_or_path = os.path.join(args.output_dir, "task_" + str(step_id - 1))

        output_dir = os.path.join(args.output_dir, "task_{}".format(step_id))
        data_dir = os.path.join(args.data_dir, "task_{}".format(step_id))
        train_and_eval(args, labels, num_labels, pad_token_label_id, model_name_or_path,
                       output_dir, data_dir, step_id)

if __name__ == "__main__":
    main()
