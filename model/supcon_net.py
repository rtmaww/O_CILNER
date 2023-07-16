import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertPreTrainedModel, BertModel
from torch.nn import CrossEntropyLoss
from util.loss_extendner import SupConLoss, KdLoss, ExtendNerLoss, \
    SupConLoss_o, BceLoss, lwf_criterion, NLLLoss, ce_bft_criterion, BceLossNoKd, bce_kl
from util.gather import gather_rh, gather_kd, gather_rh_ce
import tqdm

class SupConBertModel(BertPreTrainedModel):

    def __init__(self, config, head="mlp", feat_dim=128, per_types=6):
        super().__init__(config)
        """backbone + projection head"""
        # self.feat_dim = config.feat_dim
        self.per_types = per_types
        self.feat_dim = feat_dim
        self.hidden_size = config.hidden_size
        # self.num_labels = config.num_lables
        self.bert = BertModel(config, add_pooling_layer=False)

        # Initialize weights and apply final processing
        # self.post_init()
        self.init_weights()
        if head == 'linear':
            self.head = nn.Linear(self.hidden_size, self.feat_dim)
        elif head == 'mlp':
            self.head = nn.Sequential(
                nn.Linear(self.hidden_size, self.hidden_size),
                nn.ReLU(inplace=True),
                nn.Linear(self.hidden_size, self.feat_dim)
            )
        else:
            raise NotImplementedError(
                'head not supported: {}'.format(head))



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
        mode=None,
        num_labels=None
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

        features_enc = outputs[0]   # [batch_size, seq_length, embedding_size] embedding_size=768
        features_enc_2 = features_enc.view(-1, features_enc.shape[-1])  # [batch_size * seq_length, embedding_size]
        features = F.normalize(self.head(features_enc_2), dim=1)  # [batch_size, seq_length, feat_dim]

        loss = None
        if labels is not None:
            supcon_loss_fct = SupConLoss(temperature=0.1)
            kd_loss_fct = KdLoss()
            features = (features.view(-1, self.feat_dim)).unsqueeze(1)  # [batch_size*seq_length, 1, feature_size]
            labels = labels.view(-1)  # [batch_size*seq_length]
            supcon_loss = supcon_loss_fct(features, labels)
            # s_feat, t_feat = \
            #     gather_rh(labels.view(-1), t_features.view(-1, self.feat_dim), features.view(-1, self.feat_dim),
            #               self.num_labels-self.per_types)

            if mode == "train":
                s_feat, t_feat = \
                    gather_rh(labels.view(-1), t_logits.view(-1, self.feat_dim), features.view(-1, self.feat_dim),
                              num_labels - self.per_types)
                kd_loss = kd_loss_fct(s_feat, t_feat, t=2)
                loss = supcon_loss + kd_loss
            elif mode == "dev":
                # kd_loss = kd_loss_fct(s_feat, t_feat, t=1)
                loss = supcon_loss
            elif mode == "test":
                loss = supcon_loss

        # return {
        #     "loss": loss,
        #     "features_enc": features_enc,   # [batch_size * seq_length, embedding_size]
        #     "features": features.squeeze()  # [batch_size*seq_length, feat_dim]
        # }
        return loss, features_enc, features

class CEBertModel(BertPreTrainedModel):

    def __init__(self, config, head="mlp", feat_dim=128, per_types=6):
        super().__init__(config)
        """backbone + projection head"""
        # self.feat_dim = config.feat_dim
        self.per_types = per_types
        self.feat_dim = feat_dim
        self.hidden_size = config.hidden_size
        # self.num_labels = config.num_lables
        self.bert = BertModel(config, add_pooling_layer=False)

        # Initialize weights and apply final processing
        # self.post_init()
        self.init_weights()
        if head == 'linear':
            self.head = nn.Linear(self.hidden_size, self.feat_dim)
        elif head == 'mlp':
            self.head = nn.Sequential(
                nn.Linear(self.hidden_size, self.hidden_size),
                nn.ReLU(inplace=True),
                nn.Linear(self.hidden_size, self.feat_dim)
            )
        else:
            raise NotImplementedError(
                'head not supported: {}'.format(head))



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
        mode=None,
        num_labels=None
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

        features_enc = outputs[0]   # [batch_size, seq_length, embedding_size] embedding_size=768
        features_enc_2 = features_enc.view(-1, features_enc.shape[-1])  # [batch_size * seq_length, embedding_size]
        features = F.normalize(self.head(features_enc_2), dim=1)  # [batch_size, seq_length, feat_dim]

        loss = None
        if labels is not None:
            supcon_loss_fct = SupConLoss(temperature=0.1)
            kd_loss_fct = KdLoss()
            ce_loss_fct = CrossEntropyLoss()
            features = (features.view(-1, self.feat_dim)).unsqueeze(1)  # [batch_size*seq_length, 1, feature_size]
            labels = labels.view(-1)  # [batch_size*seq_length]
            supcon_loss = supcon_loss_fct(features, labels)
            logits = features.view(-1, self.feat_dim)
            t_logits = t_logits.view(-1, self.feat_dim)
            labels_new, s_new, s_old, teacher = gather_rh_ce(labels, t_logits, logits, num_labels-self.per_types)
            ce_loss = ce_loss_fct(s_new, labels_new)
            kd_loss = kd_loss_fct(s_old, teacher, t=2)
            # s_feat, t_feat = \
            #     gather_rh(labels.view(-1), t_features.view(-1, self.feat_dim), features.view(-1, self.feat_dim),
            #               self.num_labels-self.per_types)

            if mode == "train":
                loss = supcon_loss
            elif mode == "dev":
                # kd_loss = kd_loss_fct(s_feat, t_feat, t=1)
                loss = supcon_loss
            elif mode == "test":
                loss = supcon_loss

        # return {
        #     "loss": loss,
        #     "features_enc": features_enc,   # [batch_size * seq_length, embedding_size]
        #     "features": features.squeeze()  # [batch_size*seq_length, feat_dim]
        # }
        return loss, features_enc, features

class SupConTeacherBertModel(BertPreTrainedModel):

    def __init__(self, config, head="mlp"):
        super().__init__(config)
        """backbone + projection head"""
        self.feat_dim = 128
        self.hidden_size = config.hidden_size

        self.bert = BertModel(config, add_pooling_layer=False)

        # Initialize weights and apply final processing
        # self.post_init()
        self.init_weights()
        if head == 'linear':
            self.head = nn.Linear(self.hidden_size, self.feat_dim)
        elif head == 'mlp':
            self.head = nn.Sequential(
                nn.Linear(self.hidden_size, self.hidden_size),
                nn.ReLU(inplace=True),
                nn.Linear(self.hidden_size, self.feat_dim)
            )
        else:
            raise NotImplementedError(
                'head not supported: {}'.format(head))



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
        t_features=None,
        mode=None
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

        features_enc = outputs[0]   # [batch_size, seq_length, embedding_size] embedding_size=768
        features = F.normalize(self.head(features_enc), dim=2)  # [batch_size, seq_length, feature_size]

        # return {
        #     "features_enc": features_enc,
        #     "features": features
        # }
        return features, features_enc

class SupConBeginBertModel(BertPreTrainedModel):
    _keys_to_ignore_on_load_unexpected = [r"pooler"]

    def __init__(self, config, head="mlp"):
        super().__init__(config)
        """backbone + projection head"""
        #self.num_labels = config.num_lables
        self.per_types = 6
        self.feat_dim = 128
        self.hidden_size = config.hidden_size
        self.bert = BertModel(config, add_pooling_layer=False)

        # Initialize weights and apply final processing
        # self.post_init()
        self.init_weights()
        if head == 'linear':
            self.head = nn.Linear(self.hidden_size, self.feat_dim)
        elif head == 'mlp':
            self.head = nn.Sequential(
                nn.Linear(self.hidden_size, self.hidden_size),
                nn.ReLU(inplace=True),
                nn.Linear(self.hidden_size, self.feat_dim)
            )
        else:
            raise NotImplementedError(
                'head not supported: {}'.format(head))



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
        mode=None
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

        features_enc = outputs[0]   # [batch_size, seq_length, embedding_size] embedding_size=768
        features_enc2 = features_enc.view(-1, features_enc.shape[-1])  # [batch_size * seq_length, embedding_size]
        features = F.normalize(self.head(features_enc2), dim=1)  # [batch_size*seq_length, feat_dim]

        loss = None
        if labels is not None:
            supcon_loss_fct = SupConLoss_o(temperature=0.1)
            kd_loss_fct = KdLoss()
            bce_loss_fct = BceLoss()
            features = (features.view(-1, self.feat_dim)).unsqueeze(1)  # [batch_size*seq_length, 1, feature_size]

            labels = labels.view(-1)  # [batch_size*seq_length]
            supcon_loss = supcon_loss_fct(features, labels)
            # features = features.view(-1, features.shape[-1])
            # ce_loss_fct = CrossEntropyLoss()
            # ce_loss = ce_loss_fct(features, labels)
            # bce_loss = bce_loss_fct(features, labels)
            loss = supcon_loss

        # return {
        #     "loss": loss,
        #     "features_enc": features_enc,   # [batch_size , seq_length, embedding_size]
        #     "features": features.squeeze()  # [batch_size*seq_length, feat_dim]
        # }
        return loss, features_enc, features

class CEBeginBertModel(BertPreTrainedModel):
    _keys_to_ignore_on_load_unexpected = [r"pooler"]

    def __init__(self, config, head="mlp"):
        super().__init__(config)
        """backbone + projection head"""
        #self.num_labels = config.num_lables
        self.per_types = 6
        self.feat_dim = 128
        self.hidden_size = config.hidden_size
        self.bert = BertModel(config, add_pooling_layer=False)

        # Initialize weights and apply final processing
        # self.post_init()
        self.init_weights()
        if head == 'linear':
            self.head = nn.Linear(self.hidden_size, self.feat_dim)
        elif head == 'mlp':
            self.head = nn.Sequential(
                nn.Linear(self.hidden_size, self.hidden_size),
                nn.ReLU(inplace=True),
                nn.Linear(self.hidden_size, self.feat_dim)
            )
        else:
            raise NotImplementedError(
                'head not supported: {}'.format(head))



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
        num_labels=None,
        mode=None
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

        features_enc = outputs[0]   # [batch_size, seq_length, embedding_size] embedding_size=768
        loss = None
        if mode != "train":
            return loss, features_enc

        features_enc2 = features_enc.view(-1, features_enc.shape[-1])  # [batch_size * seq_length, embedding_size]
        classifier = nn.Linear(self.hidden_size, num_labels)
        classifier.to(device=features_enc2.device)
        logits = F.normalize(classifier(features_enc2), dim=1)  # [batch_size*seq_length, feat_dim]
        features = F.normalize(self.head(features_enc2), dim=1)

        if labels is not None:
            supcon_loss_fct = SupConLoss_o(temperature=0.1)
            kd_loss_fct = KdLoss()
            bce_loss_fct = BceLoss()
            features = (features.view(-1, self.feat_dim)).unsqueeze(1)  # [batch_size*seq_length, 1, feature_size]

            labels = labels.view(-1)  # [batch_size*seq_length]
            supcon_loss = supcon_loss_fct(features, labels)
            logits = logits.view(-1, logits.shape[-1])
            ce_loss_fct = CrossEntropyLoss()
            ce_loss = ce_loss_fct(logits, labels)
            # bce_loss = bce_loss_fct(features, labels)
            loss = ce_loss+supcon_loss

        # return {
        #     "loss": loss,
        #     "features_enc": features_enc,   # [batch_size , seq_length, embedding_size]
        #     "features": features.squeeze()  # [batch_size*seq_length, feat_dim]
        # }
        return loss, features_enc, features

class BCEBeginBertModel(BertPreTrainedModel):
    _keys_to_ignore_on_load_unexpected = [r"pooler"]

    def __init__(self, config, head="mlp"):
        super().__init__(config)
        """backbone + projection head"""
        #self.num_labels = config.num_lables
        self.per_types = 6
        self.feat_dim = 128
        self.hidden_size = config.hidden_size
        self.bert = BertModel(config, add_pooling_layer=False)

        # Initialize weights and apply final processing
        # self.post_init()
        self.init_weights()
        if head == 'linear':
            self.head = nn.Linear(self.hidden_size, self.feat_dim)
        elif head == 'mlp':
            self.head = nn.Sequential(
                nn.Linear(self.hidden_size, self.hidden_size),
                nn.ReLU(inplace=True),
                nn.Linear(self.hidden_size, self.feat_dim)
            )
        else:
            raise NotImplementedError(
                'head not supported: {}'.format(head))



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
        num_labels=None,
        mode=None
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

        features_enc = outputs[0]   # [batch_size, seq_length, embedding_size] embedding_size=768
        loss = None
        if mode != "train":
            return loss, features_enc

        features_enc2 = features_enc.view(-1, features_enc.shape[-1])  # [batch_size * seq_length, embedding_size]
        classifier = nn.Linear(self.hidden_size, num_labels)
        classifier.to(device=features_enc2.device)
        logits = F.normalize(classifier(features_enc2), dim=1)  # [batch_size*seq_length, feat_dim]
        features = F.normalize(self.head(features_enc2), dim=1)

        if labels is not None:
            supcon_loss_fct = SupConLoss_o(temperature=0.1)
            kd_loss_fct = KdLoss()
            bce_loss_fct = BceLoss()
            features = (features.view(-1, self.feat_dim)).unsqueeze(1)  # [batch_size*seq_length, 1, feature_size]
            labels = labels.view(-1)  # [batch_size*seq_length]
            supcon_loss = supcon_loss_fct(features, labels)
            logits = logits.view(-1, logits.shape[-1])
            # ce_loss_fct = CrossEntropyLoss()
            # ce_loss = ce_loss_fct(logits, labels)
            bce_loss = bce_loss_fct(logits, labels, num_labels)
            loss = bce_loss + supcon_loss

        # return {
        #     "loss": loss,
        #     "features_enc": features_enc,   # [batch_size , seq_length, embedding_size]
        #     "features": features.squeeze()  # [batch_size*seq_length, feat_dim]
        # }
        return loss, features_enc, features

class MyKDBertModel(BertPreTrainedModel):

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
        if requires_grad is False:
            for param in self.bert.parameters():
                param.requires_grad = False
            # if self.num_labels - 1 > self.per_types:
            #     for param in self.classifier.parameters():
            #         param.requires_grad = False
        # if head == 'linear':
        #     self.head = nn.Linear(self.hidden_size, self.feat_dim)
        # elif head == 'mlp':
        #     self.head = nn.Sequential(
        #         nn.Linear(self.hidden_size, self.hidden_size),
        #         nn.ReLU(inplace=True),
        #         nn.Linear(self.hidden_size, self.feat_dim)
        #     )
        # else:
        #     raise NotImplementedError(
        #         'head not supported: {}'.format(head))


    def new_classifier(self):
        weight = self.classifier.weight.data
        bias = self.classifier.bias.data
        # new_num_labels = self.num_labels+new_num_labels
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
        t_features=None,
        mode=None,
        loss_name=None
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

        features_enc = outputs[0]   # [batch_size, seq_length, embedding_size] embedding_size=768
        # features_enc_2 = features_enc.view(-1, features_enc.shape[-1])  # [batch_size * seq_length, embedding_size]
        # features = F.normalize(self.head(features_enc_2), dim=1)  # [batch_size, seq_length, feat_dim]

        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        loss = None
        if mode != "train":
            return loss, features_enc, logits

        if labels is not None:
            # supcon_loss_fct = SupConLoss(temperature=0.1)
            # supcon_o_loss_fct = SupConLoss_o(temperature=0.1)
            kd_loss_fct = KdLoss()
            ce_loss_fct = CrossEntropyLoss()
            nll_loss_fct = NLLLoss()
            bce_loss_fct = BceLoss()

            # features = (features.view(-1, features.shape[-1])).unsqueeze(1)  # [batch_size*seq_length, 1, feature_size]
            labels = labels.view(-1)  # [batch_size*seq_length]
            logits = logits.view(-1, self.num_labels)
            # supcon_loss = supcon_loss_fct(features, labels)
            # supcon_o_loss = supcon_o_loss_fct(features, labels)

            if self.num_labels-1 == self.per_types:
                ce_loss = ce_loss_fct(logits, labels)
                # bce_loss = bce_loss_fct(logits, labels, self.num_labels)
                if loss_name == "ce":
                    loss = ce_loss
                elif loss_name == "lwf":
                    # loss = lwf_criterion(logits, labels, T=2)
                    loss = ce_loss
                elif loss_name == "ce_bft":
                    loss = ce_loss


            elif self.num_labels > self.per_types:
                # print(t_logits.size())
                t_logits = t_logits.view(-1, t_logits.shape[-1])
                if loss_name == "ce_bft":
                    loss = ce_bft_criterion(logits, labels, t_logits, self.num_labels - self.per_types, T=2)
                    print(""""ce_bft:{}""".format(loss))
                    return loss, features_enc, logits
                # labels, student_new, student_old, teacher = gather_rh_ce(
                #     labels, t_logits, logits, self.num_labels - self.per_types)

                labels_new, student_new, student_old, teacher = \
                    gather_kd(labels, t_logits, logits,
                              self.num_labels - self.per_types, self.num_labels)


                if labels_new.shape[0] != 0:
                    ce_loss = nll_loss_fct(torch.log(student_new), labels_new)
                    # bce_loss = bce_loss_fct(student_new, labels, self.num_labels)
                else:
                    ce_loss = 0.
                    # bce_loss = 0.
                kd_loss = kd_loss_fct(student_old, teacher, t=2)

                if loss_name == "ce":
                    loss = ce_loss+kd_loss
                elif loss_name == "lwf":
                    # loss = lwf_criterion(logits, labels, t_logits, T=2)
                    # loss = loss
                    loss = ce_loss+kd_loss
                elif loss_name == "ce_bft":
                    loss = ce_bft_criterion(logits, labels, t_logits,
                                            self.num_labels - self.per_types, T=2)
                # elif loss_name == "bce":
                #     bce_loss = bce_loss_fct(logits, labels, self.num_labels, t_logits)
                #     loss = bce_loss+kd_loss

        return loss, features_enc, logits

class MyBertModel(BertPreTrainedModel):

    def __init__(self, config, head="mlp", feat_dim=128, per_types=6, mode="train"):
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
            self.head = nn.Linear(self.hidden_size, self.feat_dim)
        elif head == 'mlp':
            self.head = nn.Sequential(
                nn.Linear(self.hidden_size, self.hidden_size),
                nn.ReLU(inplace=True),
                nn.Linear(self.hidden_size, self.feat_dim)
            )
        else:
            raise NotImplementedError(
                'head not supported: {}'.format(head))


    def new_classifier(self):
        weight = self.classifier.weight.data
        bias = self.classifier.bias.data
        # new_num_labels = self.num_labels+new_num_labels
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
        t_features=None,
        mode=None,
        loss_name=None
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

        features_enc = outputs[0]   # [batch_size, seq_length, embedding_size] embedding_size=768
        features_enc_2 = features_enc.view(-1, features_enc.shape[-1])  # [batch_size * seq_length, embedding_size]
        features = F.normalize(self.head(features_enc_2), dim=1)  # [batch_size, seq_length, feat_dim]

        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        loss = None
        if mode != "train":
            return loss, features_enc, features, logits

        if labels is not None:
            supcon_loss_fct = SupConLoss(temperature=0.1)
            supcon_o_loss_fct = SupConLoss_o(temperature=0.1)
            kd_loss_fct = KdLoss()
            ce_loss_fct = CrossEntropyLoss()
            bce_loss_fct = BceLoss()

            features = (features.view(-1, features.shape[-1])).unsqueeze(1)  # [batch_size*seq_length, 1, feature_size]
            labels = labels.view(-1)  # [batch_size*seq_length]
            logits = logits.view(-1, self.num_labels)
            supcon_loss = supcon_loss_fct(features, labels)
            supcon_o_loss = supcon_o_loss_fct(features, labels)

            if self.num_labels-1 == self.per_types:
                ce_loss = ce_loss_fct(logits, labels)
                bce_loss = bce_loss_fct(logits, labels, self.num_labels)
                if loss_name == "supcon":
                    loss = supcon_loss
                elif loss_name == "supcon_nokd":
                    loss = supcon_loss
                elif loss_name == "supcon_o":
                    loss = supcon_o_loss
                elif loss_name == "ce":
                    loss = ce_loss
                elif loss_name == "bce":
                    loss = bce_loss
                elif loss_name == "supcon_ce":
                    loss = supcon_loss + ce_loss
                elif loss_name == "supcon_bce":
                    loss = supcon_loss + bce_loss

            elif self.num_labels > self.per_types:
                # print(t_logits.size())
                t_logits = t_logits.view(-1, t_logits.shape[-1])

                # labels, student_new, student_old, teacher = gather_rh_ce(
                #     labels, t_logits, logits, self.num_labels - self.per_types)

                labels, student_new, student_old, teacher = \
                    gather_kd(labels, t_logits, logits,
                              self.num_labels - self.per_types, self.num_labels)


                if labels.shape[0] != 0:
                    ce_loss = ce_loss_fct(student_new, labels)
                    # bce_loss = bce_loss_fct(student_new, labels, self.num_labels)
                else:
                    ce_loss = 0.
                    # bce_loss = 0.
                kd_loss = kd_loss_fct(student_old, teacher, t=2)


                if loss_name == "supcon":
                    loss = supcon_loss+kd_loss
                elif loss_name == "supcon_nokd":
                    loss = supcon_loss
                elif loss_name == "supcon_o":
                    loss = supcon_o_loss+kd_loss
                elif loss_name == "ce":
                    loss = ce_loss+kd_loss
                    # loss = loss
                elif loss_name == "bce":
                    bce_loss = bce_loss_fct(logits, labels, self.num_labels, t_logits)
                    loss = bce_loss+kd_loss
                elif loss_name == "supcon_ce":
                    loss = supcon_loss+ce_loss+kd_loss
                elif loss_name == "supcon_bce":
                    bce_loss = bce_loss_fct(logits, labels, self.num_labels, t_logits)
                    loss = supcon_loss+bce_loss

        # return {
        #     "loss": loss,
        #     "features_enc": features_enc,   # [batch_size * seq_length, embedding_size]
        #     "features": features.squeeze()  # [batch_size*seq_length, feat_dim]
        # }
        return loss, features_enc, features, logits


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


def get_augmemtation_data(features_enc, entity_top_emissions, mix_up, labels, num_old_labels):
    import random
    import numpy as np
    device = features_enc.device
    features_enc = features_enc.view(-1, features_enc.shape[-1])  #[bsz*sql, feat_dim]
    # aug_features = torch.zeros(features_enc.shape[0], mix_up, features_enc.shape[-1])
    # idx = torch.where(labels > num_old_labels-1)[0]
    idx1 = torch.randint(0, mix_up - 1, (features_enc.shape[0], mix_up)).to(device)
    idx2 = torch.randint(0, mix_up - 1, (features_enc.shape[0], mix_up)).to(device)
    mixup_idx1 = torch.gather(entity_top_emissions, -1, idx1.long())
    mixup_idx2 = torch.gather(entity_top_emissions, -1, idx2.long())
    alpha = 0.2
    # lam = np.random.beta(alpha, alpha)
    lam = np.random.beta(alpha, alpha, features_enc.shape[0]*mix_up)
    # print(lam)
    lam = torch.tensor(lam).view(-1, mix_up).unsqueeze(2).double().to(device)
    features_enc_mixup1 = features_enc[mixup_idx1]
    features_enc_mixup2 = features_enc[mixup_idx2]
    aug_features = lam * features_enc_mixup1 + (1 - lam) * features_enc_mixup2
    return aug_features.to(torch.float32).to(device)
