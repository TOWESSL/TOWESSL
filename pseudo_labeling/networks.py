import math
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.autograd import Variable
import os
from transformers import BertTokenizer, BertModel
import pickle
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pad_i=0


class Pos_model(torch.nn.Module):
    def __init__(self, word_embed_dim, output_size, vocab_size, args=None):
        super(Pos_model, self).__init__()
        self.model_name = args.model if args is not None else False
        # input size: 768 for bert
        if(args.word_embed_method=='bert'):
            self.input_size = 768
        else:
            self.input_size = word_embed_dim
        # hidden size: for RNN Tagger
        self.hidden_size = args.n_hidden
        # output size: N of Tags
        self.output_size = output_size
        
        self.layer_size=2
        self.args=args
        if(self.layer_size==1):
            self.dropout=0
        else:
            self.dropout=0.5

        # Embedding: GLOVE or BERT        
        self.word_embed_method=args.word_embed_method
        self.init_embedding = np.load(os.path.join('../../../common_data/embedding', args.embed_name))
        self.init_embedding = torch.tensor(np.float32(self.init_embedding)).to(device)
        self.bert_model = BertModel.from_pretrained('bert-base-uncased', return_dict=True)
        
        # Encoder: RNN or Transformer
        if args.rnn_method == 'LSTM':
            self.BiGRU = torch.nn.GRU(self.input_size ,self.hidden_size, num_layers=self.layer_size, bidirectional=True, batch_first=True,dropout=self.dropout)
            self.BiLSTM = torch.nn.LSTM(self.input_size ,self.hidden_size, num_layers=self.layer_size, bidirectional=True, batch_first=True,dropout=self.dropout)
        else:
            self.transFC = torch.nn.Linear(self.input_size, 512)
            self.transformerEncoderLayer = torch.nn.TransformerEncoderLayer(512, nhead=8, batch_first=True)
            self.transformerEncoder = torch.nn.TransformerEncoder(self.transformerEncoderLayer, num_layers=self.layer_size)

        # FC Layer for BIO Tagging
        if args.rnn_method == 'TRANSFORMER':
            self.fc_tar = torch.nn.Linear(512, self.output_size)
        else:
            self.fc_tar = torch.nn.Linear(self.hidden_size * 2, self.output_size)
        self.init_weight()

    def init_weight(self):
        if self.args.rnn_method != 'TRANSFORMER':
            for weights in [self.BiLSTM.weight_hh_l0, self.BiLSTM.weight_ih_l0]:
                torch.nn.init.orthogonal_(weights)
            for weights in [self.BiGRU.weight_hh_l0, self.BiGRU.weight_ih_l0]:
                torch.nn.init.orthogonal_(weights)

        # linear
        torch.nn.init.xavier_normal_(self.fc_tar.weight)
    
    def word_forward(self, batch,train_bert):    
        if(self.word_embed_method=='glove'):
            sentence, _ = batch.text
            words_embeds=self.init_embedding[sentence]
        elif(self.word_embed_method=='bert'):
            bert_ids=batch.bert_ids
            bert_mask=batch.bert_mask
            outputs = self.bert_model(bert_ids, attention_mask=bert_mask)
            #words_embeds = outputs.last_hidden_state
            if(train_bert):
                words_embeds=outputs[0]
            else:
                words_embeds=outputs[0].detach()

        return words_embeds
        #return words_embeds.detach()

    def word_aug_forward(self, batch,train_bert):
        if(self.word_embed_method=='bert'):
            bert_ids=batch.aug_bert_ids
            bert_mask=batch.aug_bert_mask
            outputs = self.bert_model(bert_ids, attention_mask=bert_mask)
            #words_embeds = outputs.last_hidden_state
            if(train_bert):
                words_embeds=outputs[0]
            else:
                words_embeds=outputs[0].detach()

        return words_embeds


    def forward(self, batch ,train_bert, aug=False):
        if(aug==True):
            sentence = self.word_aug_forward(batch,train_bert)
        else:
            sentence = self.word_forward(batch,train_bert)

        #sentence=F.dropout(sentence,p=0.5)#LOTN
        if(self.args.rnn_method=='GRU'):
            encoded, _ = self.BiGRU(sentence)
        elif self.args.rnn_method == 'LSTM':
            encoded, _ = self.BiLSTM(sentence)
        else: # transformer
            trans_emb = self.transFC(sentence)
            transformer_mask = (1 - batch.bert_mask).to(torch.bool)
            encoded = self.transformerEncoder(trans_emb, src_key_padding_mask = transformer_mask)
        
        decodedP_tar = self.fc_tar(encoded)
        log_outputP = F.log_softmax(decodedP_tar, dim=-1)
        outputP = F.softmax(decodedP_tar, dim=-1)
        return outputP,log_outputP

