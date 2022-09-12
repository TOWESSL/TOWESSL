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




class Pos_model(torch.nn.Module):
    def __init__(self, word_embed_dim, output_size, vocab_size, args=None):
        super(Pos_model, self).__init__()
        
        # input size: 768 for bert
        if(args.word_embed_method=='bert'):
            self.input_size = 768
        else:
            self.input_size = word_embed_dim
        # bio_size: hidden size for label embedding
        self.bio_size=self.input_size
        # hidden size: for RNN Labeler
        self.hidden_size = args.n_hidden
        # output size: Num of Tags + 1(0: padding)
        self.output_size = output_size
        # layer size: num of layers in RNN
        self.layer_size=args.layer_size

        self.args=args
        
        if(self.layer_size==1):
            self.dropout=0
        else:
            self.dropout=0.5

        # Networks
        # Token Encoder: GLOVE or BERT
        self.word_embed_method=args.word_embed_method

        self.bert_model = BertModel.from_pretrained('bert-base-uncased', return_dict=True)

        # Label Embedding
        self.learnable_pos_embedding=torch.nn.Embedding(num_embeddings=output_size, embedding_dim=self.bio_size)

        # RNN Sequence Labeler
        if args.rnn_method == 'LSTM':
            self.BiGRU = torch.nn.GRU(self.input_size+self.bio_size ,self.hidden_size, num_layers=self.layer_size, bidirectional=True, batch_first=True,dropout=self.dropout)
            self.BiLSTM = torch.nn.LSTM(self.input_size+self.bio_size ,self.hidden_size, num_layers=self.layer_size, bidirectional=True, batch_first=True,dropout=self.dropout)
        elif args.rnn_method == 'TRANSFORMER':
            self.transFC = torch.nn.Linear(self.input_size+self.bio_size, 512)
            self.transformerEncoderLayer = torch.nn.TransformerEncoderLayer(512, nhead=8, batch_first=True)
            self.transformerEncoder = torch.nn.TransformerEncoder(self.transformerEncoderLayer, num_layers=2)

        # Self Attention Layer
        if args.rnn_method == 'TRANSFORMER':
            self.self_attn = torch.nn.MultiheadAttention(embed_dim=512, num_heads=8, batch_first=True)
        else:
            self.self_attn = torch.nn.MultiheadAttention(embed_dim=self.hidden_size * 2, num_heads=8, batch_first=True)
        
        # Full Connect Layer for BIO tagging
        if args.rnn_method == 'TRANSFORMER':
            self.fc = torch.nn.Linear(512, self.output_size)
        else:
            self.fc = torch.nn.Linear(self.hidden_size * 2, self.output_size)

        # Positional Embedding
        if(args.pos_embed=='my'):
            self.pos_matrix=self.mypositionalencoding1d(self.bio_size,1000+1)
        elif(args.pos_embed=='ori'):
            self.pos_matrix=self.positionalencoding1d(self.bio_size,1000+1)
        elif(args.pos_embed=='tar_oth'):
            self.pos_matrix=self.tarothpositionalencoding1d(self.bio_size,2+1)

        # Sentiment Classifier
        # self.senti_classifier = nn.Linear(self.input_size ,2)
        self.senti_classifier = nn.Sequential(
            nn.Linear(self.input_size, self.input_size * 2),
            nn.Tanh(),
            nn.Linear(self.input_size * 2, self.input_size),
            nn.Tanh(),
            nn.Linear(self.input_size, 2),
        )

        self.init_weight()

    def init_weight(self):
        if self.args.rnn_method != 'TRANSFORMER':
            for weights in [self.BiLSTM.weight_hh_l0, self.BiLSTM.weight_ih_l0]:
                torch.nn.init.orthogonal_(weights)
            for weights in [self.BiGRU.weight_hh_l0, self.BiGRU.weight_ih_l0]:
                torch.nn.init.orthogonal_(weights)

        # linear
        torch.nn.init.xavier_normal_(self.learnable_pos_embedding.weight)
        torch.nn.init.xavier_normal_(self.fc.weight)
        # torch.nn.init.xavier_normal_(self.senti_classifier.weight)

    def tarothpositionalencoding1d(self,d_model, length):
        """
        :param d_model: dimension of the model
        :param length: length of positions
        :return: length*d_model position matrix
        """
        pe = torch.zeros(length, d_model)
        pe[2]=1
        return pe.to(device)
    
    def positionalencoding1d(self,d_model, length):
        """
        :param d_model: dimension of the model
        :param length: length of positions
        :return: length*d_model position matrix
        """
        if d_model % 2 != 0:
            raise ValueError("Cannot use sin/cos positional encoding with "
                             "odd dim (got dim={:d})".format(d_model))
        pe = torch.zeros(length, d_model)
        position = torch.arange(0, length).unsqueeze(1)
        div_term = torch.exp((torch.arange(0, d_model, 2, dtype=torch.float) *
                             -(math.log(10000.0) / d_model)))
        pe[:, 0::2] = torch.sin(position.float() * div_term)
        pe[:, 1::2] = torch.cos(position.float() * div_term)
        return pe.to(device)
    
    def mypositionalencoding1d(self,d_model, length):
        """
        :param d_model: dimension of the model
        :param length: length of positions
        :return: length*d_model position matrix
        """
        if d_model % 2 != 0:
            raise ValueError("Cannot use sin/cos positional encoding with "
                             "odd dim (got dim={:d})".format(d_model))
        pe = torch.zeros(length, d_model)
        position = torch.arange(0, length).unsqueeze(1)
        div_term = torch.exp((torch.arange(0, d_model, 2, dtype=torch.float) *
                             -(math.log(10000.0) / d_model)))
        pe[:, 0::2] = torch.abs(torch.sin(position.float() * div_term))
        pe[:, 1::2] = 1-torch.abs(torch.sin(position.float() * div_term))
        pe[0]=torch.tensor([-1.0]*d_model)
        return pe.to(device)

    def word_forward(self, batch,train_bert):    
        if(self.word_embed_method=='glove'):
            sentence, _ = batch.text
            # words_embeds=self.init_embedding[sentence]
            raise RuntimeError("Glove has been discarded")

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
        if(self.word_embed_method=='glove'):
            sentence, _ = batch.aug_text
            raise RuntimeError("Glove has been discarded")
            # words_embeds=self.init_embedding[sentence]
        elif(self.word_embed_method=='bert'):
            bert_ids=batch.aug_bert_ids
            bert_mask=batch.aug_bert_mask
            outputs = self.bert_model(bert_ids, attention_mask=bert_mask)
            #words_embeds = outputs.last_hidden_state
            if(train_bert):
                words_embeds=outputs[0]
            else:
                words_embeds=outputs[0].detach()
        

        return words_embeds
        
    def forward(self, batch, train_bert, aug=False, ulb=False):
        # Positional Embedding: [Batch, Sequence, input_size]
        if(self.args.pos_embed=='my' or self.args.pos_embed=='ori'):
            pos_embedding=self.pos_matrix[batch.dis]
        elif(self.args.pos_embed=='pos_learn'):
            pos_embedding=self.learnable_pos_embedding(batch.target)
        elif (self.args.pos_embed=='tar_oth'):
            pos_embedding=self.pos_matrix[batch.tar_oth]

        # Token Embedding: [Batch, Sequence, input_size:768]
        if(aug==True):
            sentence = self.word_aug_forward(batch,train_bert)
        else:
            sentence = self.word_forward(batch,train_bert)
        final_embedding=torch.cat([pos_embedding,sentence],dim=-1)

        # RNN tagger: [Batch, Sequence, hidden_size*2: 200*2]
        if(self.args.rnn_method=='GRU'):
            encoded, _ = self.BiGRU(final_embedding)
        elif(self.args.rnn_method=='LSTM'):
            encoded, _ = self.BiLSTM(final_embedding)
        elif(self.args.rnn_method=='TRANSFORMER'):
            trans_emb = self.transFC(final_embedding)
            transformer_mask = (1 - batch.bert_mask).to(torch.bool)
            encoded = self.transformerEncoder(trans_emb, src_key_padding_mask=transformer_mask)
        
        # Self Attn: Attned output:[Batch, Sequence, hidden_size*2: 200*2], Attn Weight: [Batch, Sequence, Sequence]
        attn_mask = 1 - batch.bert_mask
        attned_encoded, attn_weight = self.self_attn(encoded, encoded, encoded, key_padding_mask=attn_mask)

        # Linear Tagger
        decodedP = self.fc(attned_encoded)
        outputP = F.softmax(decodedP/self.args.tau, dim=-1)
        log_outputP=F.log_softmax(decodedP/self.args.tau,dim=-1)

        return decodedP,outputP,log_outputP
    

