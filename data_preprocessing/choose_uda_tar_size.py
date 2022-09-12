import torch
from tqdm import tqdm,trange
import copy
import pickle
import os
import numpy as np
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec
import codecs
import random
import time
import numpy as np
from numpy import linalg as la
import math
import nltk
from random import sample
from transformers import BertTokenizer

random.seed(20200202)
np.random.seed(20200202)

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1','True'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0','False'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--ds", type=str, default='14res')
parser.add_argument("--raw", type=str, default='yelp')
parser.add_argument("--aug_file", type=str, default='unlabel_tar_train.tsv')
parser.add_argument("--aug_size", type=int, default=500000)
args = parser.parse_args()

print("aug size:", args.aug_size)

if(args.raw=='amazon'):
    raw=['./common_data/amazon/senti_reviews_Electronics_5.txt']
elif args.raw == 'lap':
    raw = ['./common_data/lap/senti_amazon_lap_ulb.txt']
elif(args.raw=='yelp'):
    raw=['./common_data/yelp/senti_yelp_academic_dataset_review.txt']


tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
w2i = tokenizer.vocab
i2w = {}
for k, v in w2i.items():
    i2w[v] = k 

class train_instance:
    def __init__(self, line,writer):
        line= line.strip()
        sentence=line.split('\t')[0]
        self.senti=line.split('\t')[1]
        if self.senti == 'positive':
            self.senti = '0'
        elif self.senti == 'negtive':
            self.senti = '1'
        sentence = np.array(sentence.split(' '))
        self.s_id='semi'
        self.sentence=sentence
        self.writer=writer
        self.write_flag=False

    def check(self):
        # Choose only the sentence whose words are all in vocab
        sentence=copy.deepcopy(self.sentence)
        sentence=sentence.tolist()
        for i in sentence:
            if(i not in w2i):
                return False
        return True

    def write_instance(self):
        self.writer.write(self.s_id)
        self.writer.write('\t')
        for index,i in enumerate(self.sentence):
            self.writer.write(i)
            if(index!=self.sentence.shape[0]-1):
                self.writer.write(' ')
        self.writer.write('\t')
        self.writer.write(self.senti)
        self.writer.write('\n')

lines=[]
for i in raw:
    with codecs.open(i) as f:
        lines.extend(f.readlines())
random.shuffle(lines)
print(len(lines))
# Sample only aug_size from raw data
lines=lines[:args.aug_size]
with codecs.open("./data/%s/%s" % (args.ds,args.aug_file), "w") as writer:
    writer.write("s_id\tsentence\tsentiment\n")
    for line_index in trange(len(lines)):
        line=lines[line_index]
        ti=train_instance(line,writer)
        # if(ti.check()==True):
        ti.write_instance()

"""
unsupervised data for specific dataset:
only sampling from ulb data, not handling lb data.
s_id    sentence    sentiment(pos|neg)
"""