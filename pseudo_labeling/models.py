import nltk
import math
import copy
from torch.nn.modules.container import Sequential
from tqdm import tqdm,trange
import os
import numpy as np
import networks
import torch
import time
from torch import nn
import train
from sklearn.feature_extraction.text import CountVectorizer  
from sklearn.feature_extraction.text import TfidfTransformer
from transformers import BertTokenizer, BertModel
import torchtext.data as data
from utils import * 

pad_i=0
tag2id = {'B': 1, 'I': 2, 'O': 3 ,'X': 4, 'SEP':5, 'CLS': 6}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



class NeuralTagger():  # Neural network method
    def __init__(self):
        self.word_embed_dim = 300
        self.hidden_size = 128
        self.vocab_size = 100
        self.output_size = 3
        pass

    def train_from_data(self, train_raw_data,test_raw_data, W, word2index,index2word, args):
        self.word_embed_dim = 300
        self.hidden_size = args.n_hidden
        self.vocab_size = 21111
        self.output_size = len(tag2id)+1

        if args.model == 'Pos_model':
            self.tagger = networks.Pos_model(self.word_embed_dim, self.output_size, self.vocab_size, args)
        else:
            print("model name not found")
            exit(-1)

        # W = torch.from_numpy(W)

        train_texts, train_targets = train_raw_data
        train_iter, dev_iter = create_labeled_dataset(train_texts, train_targets, args, train=True)

        test_texts, test_targets = test_raw_data
        test_iter = create_labeled_dataset(test_texts, test_targets, args, train=False)

        print('---train---')
        train.train(self.tagger, train_iter, dev_iter,test_iter,W,word2index,index2word, args=args)

    def make_pseu_target(self, test_raw_data, senti,  W, word2index, index2word, args):
        self.word_embed_dim = W.shape[1]
        self.hidden_size = args.n_hidden
        self.vocab_size = len(W)
        self.output_size = len(tag2id)+1
        if args.model == 'Pos_model':
            self.tagger = networks.Pos_model(self.word_embed_dim, self.output_size, self.vocab_size, args)
        else:
            print("model name not found")
            exit(-1)

        if(args.random_make_pseu_target==False):
            self.tagger.load_state_dict(torch.load(os.path.join("backup", args.test_model)))
        self.tagger.to(device)
        W = torch.from_numpy(W)
        


        BERT_IDS=data.Field(sequential=True, use_vocab=False, pad_token=0, batch_first=True)
        BERT_MASK=data.Field(sequential=True, use_vocab=False, pad_token=0, batch_first=True)
        TEXT = data.Field(sequential=True, use_vocab=False, pad_token=0,  batch_first=True, include_lengths=True)
        SENTI = data.Field(sequential=False, use_vocab=False, batch_first=True)

        fields = [('bert_ids',BERT_IDS),('bert_mask',BERT_MASK),('text', TEXT), ('senti', SENTI)]

        test_bert_ids_dic,test_bert_mask_dic, _ = numericalize_bert(test_raw_data)
        test_data = [[test_bert_ids_dic[text],
                      test_bert_mask_dic[text],
                      numericalize(text, word2index),
                      sent,
                      ] for text, sent in zip(test_raw_data, senti)]
        
        test_dataset = ToweDataset(fields, test_data,'unlabel_tar')
        test_iter = data.Iterator(test_dataset, batch_size=args.eval_bs, shuffle=True, sort_within_batch=True,
                                  repeat=False,
                                  device=device)
        with torch.no_grad():
            if(args.random_make_pseu_target==False):
                label_text,label_target,senti_list = train.make_pseu_target(self.tagger, test_iter, W,word2index,index2word,args=args)
            
        
        pseudo_num=0
        with open('./data/%s/unsplit_unlabel_train_%d.tsv'%(args.ds,args.cur_run_times),'w') as f:
            for i in trange(len(label_text)):
                lte=' '.join(label_text[i][1:-1])

                w_flag=False
                for j in range(len(label_target[i])):
                    if('\\B' in label_target[i][j] or '\\I' in label_target[i][j]):
                        w_flag=True
                if(w_flag==False):continue

                lta=' '.join(label_target[i][1:-1])
                f.write('semi'+'\t')
                f.write(lte+'\t')
                f.write(lta+'\t')
                f.write(str(senti_list[i])+'\n')
                pseudo_num+=1
        print('pseudo labels made. final pseudo size %d'%pseudo_num)
