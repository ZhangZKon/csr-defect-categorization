import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from utils.myLogger import logging

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0):
        super().__init__()
        self.gamma = gamma

    def forward(self, logits, labels):
        ce_loss = F.cross_entropy(logits, labels, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()

NUM_EDIT_ACTIONS = 5  # PAD=0, EQUAL=1, INSERT=2, DELETE=3, REPLACE=4

class EditActionEmbedding(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.embedding = nn.Embedding(NUM_EDIT_ACTIONS, hidden_size, padding_idx=0)
        nn.init.zeros_(self.embedding.weight)

    def forward(self, edit_action_ids):
        return self.embedding(edit_action_ids)

class RobertaClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config):
        super(RobertaClassificationHead, self).__init__()
        self.dense = nn.Linear(config.hidden_size * 2, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features):
        x = features[:, 0, :]  
        x = x.reshape(-1, x.size(-1) * 2)  

        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x

class Model(nn.Module):
    """
        Build Seqence-to-Sequence.

        Parameters:

        * `encoder`- encoder of seq2seq model. e.g. roberta
        * `decoder`- decoder of seq2seq model. e.g. transformer
        * `config`- configuration of encoder model.
        * `beam_size`- beam size for beam search.
        * `max_length`- max length of target for beam search.
        * `sos_id`- start of symbol ids in target for beam search.
        * `eos_id`- end of symbol ids in target for beam search.
    """

    def __init__(self, encoder, config):
        super(Model, self).__init__()
        self.encoder = encoder
        self.classifier = RobertaClassificationHead(config)
        self.config = config
        self.sm = nn.Softmax(dim=-1)

        self.edit_action_embedding = EditActionEmbedding(config.hidden_size)
        self.focal_loss = FocalLoss(gamma=2.0)

    def forward(self, bug_ids, bug_mask,
                clean_ids, clean_mask,
                labels, edit_action_ids=None):
        bs, l = bug_ids.size()
        
        if edit_action_ids is not None:
            bug_base_emb = self.encoder.embeddings.word_embeddings(bug_ids)
            edit_emb = self.edit_action_embedding(edit_action_ids)
            bug_inputs_embeds = bug_base_emb + edit_emb
            bug_outputs = self.encoder(inputs_embeds=bug_inputs_embeds, attention_mask=bug_mask)[0]
        
            clean_outputs = self.encoder(input_ids=clean_ids, attention_mask=clean_mask)[0]
        
            outputs = torch.stack([bug_outputs, clean_outputs], dim=1).view(bs * 2, l, -1)
        else:
            inputs_ids = torch.cat((bug_ids.unsqueeze(1), clean_ids.unsqueeze(1)), 1).view(bs * 2, l)
            attn_mask = torch.cat((bug_mask.unsqueeze(1), clean_mask.unsqueeze(1)), 1).view(bs * 2, l)
            outputs = self.encoder(input_ids=inputs_ids, attention_mask=attn_mask)[0]
        
        cls_logits = self.classifier(outputs)
        cls_prob = self.sm(cls_logits)
        
        # CE -> Focal
        cls_loss = self.focal_loss(cls_logits, labels)
        
        return cls_loss, cls_prob
