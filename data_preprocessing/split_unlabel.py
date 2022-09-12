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
parser.add_argument("--cur_run_times", type=int, default=0)
parser.add_argument("--split_file", type=str, default='unsplit_unlabel_train_')

args = parser.parse_args()
split_file=os.path.join('./data',args.ds,args.split_file+str(args.cur_run_times)+'.tsv')

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
w2i = tokenizer.vocab
i2w = {}
for k, v in w2i.items():
    i2w[v] = k 
print('voc_size',len(i2w))

class train_instance:
    def __init__(self, line,writer):
        sentence= line.strip()
        self.s_id='semi'
        self.origin_sentence=sentence.split('\t')[1]
        self.origin_tags=sentence.split('\t')[2]
        self.origin_senti = sentence.split('\t')[3]

        self.sentence=[]
        self.tags=[]
        self.senti = []
        self.writer=writer
        self.write_flag=False
    
    def split_tags(self):
        tags=copy.deepcopy(self.origin_tags)
        tags=tags.split(' ')
        t_list=[]
        for i in range(len(tags)):
            if('\\B' in tags[i]):
                t_list.append(i)
        for i in range(len(t_list)):
            self.sentence.append(self.origin_sentence)
            t=copy.deepcopy(self.origin_tags)
            t=t.split(' ')
            no=[]
            for j in range(0,len(t)):
                if(j==t_list[i]):
                    no.append(j)
                    for k in range(j+1,len(t)):
                        if('\\O' not in t[k]):
                            no.append(k)
                        else:
                            break
            for j in range(0,len(t)):
                if(j not in no):
                    t_w=t[j].split('\\')[0]
                    t[j]=t_w+'\\O'
            self.tags.append(' '.join(t))
            self.senti.append(self.origin_senti)


    def write_instance(self):
        for s_num in range(len(self.sentence)):
            self.writer.write('semi')
            self.writer.write('\t')
            self.writer.write(self.sentence[s_num])
            self.writer.write('\t')
            self.writer.write(self.tags[s_num])
            self.writer.write('\t')
            self.writer.write(self.senti[s_num])
            self.writer.write('\n')

lines=[]
with codecs.open(split_file) as f:
    lines.extend(f.readlines())
lines=lines[1:]
random.shuffle(lines)

with codecs.open(os.path.join('./data',args.ds,'unlabel_train_'+str(args.cur_run_times)+'.tsv'), "w") as writer:
    writer.write("s_id\tsentencei\ttarget_tags\tsenti\n")
    for line_index in trange(len(lines)):
        line=lines[line_index]
        ti=train_instance(line,writer)
        ti.split_tags()
        ti.write_instance()
