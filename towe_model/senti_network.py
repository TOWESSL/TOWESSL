import math
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.autograd import Variable
import os
import pickle
from transformers import BertTokenizer, BertModel
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pad_i=0

class Senti_model(torch.nn.Module):
    def __init__(self):
        super(Senti_model, self).__init__()
        self.bert_model = BertModel.from_pretrained(
            "bert-base-uncased"
        )

        # set the number of features our encoder model will return...
        self.encoder_features = 768
        # Tokenizer
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

        # Attn-LSTM
        self.BiLSTM = torch.nn.LSTM(self.encoder_features,300, num_layers=2, bidirectional=True, batch_first=True,dropout=0.5)
        self.fc_senti = torch.nn.Linear(600, 2) 
    
    def init_weight(self):
        for weights in [self.BiLSTM.weight_hh_l0, self.BiLSTM.weight_ih_l0]:
            torch.nn.init.orthogonal_(weights)
        torch.nn.init.xavier_normal_(self.fc_senti.weight)

    def attention_net(self, lstm_output, target_state):
        hidden = target_state.squeeze(0)
        attn_weights = torch.bmm(lstm_output, hidden.unsqueeze(2)).squeeze(2)
        soft_attn_weights = F.softmax(attn_weights, 1)
        new_hidden_state = torch.bmm(lstm_output.transpose(1, 2), soft_attn_weights.unsqueeze(2)).squeeze(2)
        return new_hidden_state, soft_attn_weights

    def forward(self, batch, train_bert=True):
        bert_ids=batch.bert_ids
        bert_mask=batch.bert_mask
        outputs = self.bert_model(bert_ids, attention_mask=bert_mask, return_dict=True)
        sentence = outputs.last_hidden_state
        final_embedding=sentence

        encoded, (final_hidden_state,_) = self.BiLSTM(final_embedding)

        # final_h => reshape + encoded => ATTN => senti_encoded
        final_hidden_state=final_hidden_state.view(2,2,final_hidden_state.shape[1],final_hidden_state.shape[2])[-1].permute(1,0,2)
        final_hidden_state=final_hidden_state.reshape(1,final_hidden_state.shape[0],-1)
        senti_encode,senti_attention = self.attention_net(encoded, final_hidden_state)

        logits = self.fc_senti(senti_encode)
        return logits, senti_attention.detach()


