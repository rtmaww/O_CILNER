import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertPreTrainedModel, BertModel
from torch.nn import CrossEntropyLoss
from util.loss_extendner import SupConLoss, KdLoss, ExtendNerLoss, \
    SupConLoss_o, BceLoss, lwf_criterion, NLLLoss, ce_bft_criterion, BceLossNoKd, bce_kl
from util.gather import gather_rh, gather_kd, gather_rh_ce
import tqdm


class MySftBertModel(BertPreTrainedModel):

    def __init__(self, config, head="mlp", feat_dim=128, per_types=6, mode="train", requires_grad=True):
        super().__init__(config)
        """backbone + projection head"""
        # self.feat_dim = config.feat_dim
        self.per_types = per_types
        self.feat_dim = feat_dim
        self.hidden_size = config.hidden_size
        self.num_labels = config.num_labels
        self.bert = BertModel(config, add_pooling_layer=False)

        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        if mode == "train":
            if self.num_labels-1 > self.per_types:
                self.classifier = nn.Linear(config.hidden_size, config.num_labels - self.per_types)
            else:
                self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        else:
            self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        # Initialize weights and apply final processing
        # self.post_init()
        self.init_weights()
        if head == 'linear':
            self.head = nn.Linear(self.hidden_size, self.hidden_size)
        elif head == 'mlp':
            self.head = nn.Sequential(
                nn.Linear(self.hidden_size, self.hidden_size),
                nn.ReLU(inplace=True),
                nn.Linear(self.hidden_size, self.feat_dim)
            )
        else:
            raise NotImplementedError(
                'head not supported: {}'.format(head))
        if requires_grad is False:
            for param in self.bert.parameters():
                param.requires_grad = False
            for param in self.head.parameters():
                param.requires_grad = False

    def new_classifier(self):
        weight = self.classifier.weight.data
        bias = self.classifier.bias.data
        new_cls = nn.Linear(self.hidden_size, self.num_labels)
        new_cls.weight.data[:self.num_labels-self.per_types] = weight
        new_cls.bias.data[:self.num_labels-self.per_types] = bias
        self.classifier = new_cls
        
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        t_logits=None,
        top_emissions=None,
        negative_top_emissions=None,
        mode=None,
        loss_name=None,
        pseudo_labels=None,
        entity_top_emissions=None,
        topk_th=False,
        o_weight=None
    ):
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the token classification loss. Indices should be in `[0, ..., config.num_labels - 1]`.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        features_enc = outputs[0]   # [batch_size, seq_length, embedding_size]
        features = F.normalize(self.head(features_enc.view(-1, self.hidden_size)), dim=1)
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        loss = None
        if mode != "train":
            return loss, features_enc, features, logits

        if labels is not None:
            supcon_loss_fct = SupConLoss(temperature=0.1, topk_th=topk_th)
            supcon_o_loss_fct = SupConLoss_o(temperature=0.1, topk_th=topk_th)
            kd_loss_fct = KdLoss()
            ce_loss_fct = CrossEntropyLoss()
            bce_loss_fct = BceLoss(o_weight=o_weight)

            if pseudo_labels is None:
                pseudo_labels = labels
            features = features.unsqueeze(1)  # [batch_size*seq_length, 1, embedding_size]
            labels = labels.view(-1)  # [batch_size*seq_length]
            pseudo_labels = pseudo_labels.view(-1)
            logits = logits.view(-1, self.num_labels)
            if loss_name == "supcon" or loss_name == "supcon_ce" or loss_name == "supcon_bce":
                supcon_loss = supcon_loss_fct(features, pseudo_labels,
                                              entity_topk=entity_top_emissions)
            if loss_name == "supcon_o" or loss_name == "supcon_o_ce" or loss_name == "supcon_o_bce":
                supcon_o_loss = supcon_o_loss_fct(features, pseudo_labels, top_emissions,
                                                  negative_top_emissions,
                                                  entity_topk=entity_top_emissions)

            if self.num_labels-1 == self.per_types:
                ce_loss = ce_loss_fct(logits, labels)
                bce_loss = bce_loss_fct(logits, labels, self.num_labels)
                if loss_name == "supcon":
                    loss = supcon_loss
                elif loss_name == "supcon_o":
                    loss = supcon_o_loss
                elif loss_name == "supcon_o_ce":
                    loss = supcon_o_loss+ce_loss
                elif loss_name == "supcon_o_bce":
                    loss = supcon_o_loss + bce_loss
                elif loss_name == "ce":
                    loss = ce_loss
                elif loss_name == "bce_o":
                    bce_loss = bce_loss_fct(logits, labels, self.num_labels, cal_O=True)
                    loss = bce_loss
                elif loss_name == "supcon_ce":
                    loss = supcon_loss + ce_loss
                elif loss_name == "supcon_bce":
                    loss = supcon_loss + bce_loss

            elif self.num_labels > self.per_types:
                # print(t_logits.size())
                if t_logits is not None:
                    t_logits = t_logits.view(-1, t_logits.shape[-1])

                    labels_new, student_new, s_logits, old_logits = gather_rh_ce(
                        labels, t_logits, logits, self.num_labels - self.per_types)
                    
                    if labels.shape[0] != 0:
                        ce_loss = ce_loss_fct(student_new, labels_new)
                    else:
                        ce_loss = 0.
                    kd_loss = kd_loss_fct(s_logits, old_logits, t=2)

                if loss_name == "supcon":
                    loss = supcon_loss+kd_loss
                elif loss_name == "supcon_nokd":
                    loss = supcon_loss
                elif loss_name == "supcon_o":
                    loss = supcon_o_loss+kd_loss
                elif loss_name == "supcon_o_ce":
                    loss = supcon_o_loss+ce_loss+kd_loss
                elif loss_name == "supcon_o_bce":
                    bce_loss = bce_loss_fct(logits, labels, self.num_labels, t_logits)
                    loss = supcon_o_loss + bce_loss
                if loss_name == "ce":
                    loss = ce_loss+kd_loss
                elif loss_name == "bce_o":
                    bce_loss = bce_loss_fct(logits, labels, self.num_labels, t_logits, cal_O=True)
                    loss = bce_loss
                elif loss_name == "supcon_ce":
                    # kd_loss = kd_loss_fct(s_logits, t_logits, t=2)
                    loss = supcon_loss+ce_loss+kd_loss
                elif loss_name == "supcon_bce":
                    bce_loss = bce_loss_fct(logits, labels, self.num_labels, t_logits)
                    loss = supcon_loss+bce_loss

        return loss, features_enc, features, logits
