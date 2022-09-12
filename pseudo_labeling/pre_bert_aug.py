import os 
import random
from urllib.parse import quote 
import numpy as np
from tqdm import trange
from utils import * 
from itertools import chain 
from copy import deepcopy
import csv
import nltk
from transformers import BertTokenizer, BertModel
from nltk.corpus import wordnet

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
parser.add_argument("--seed", type=int, default=100)
parser.add_argument("--cur_run_times", type=int, default=1)
parser.add_argument("--aug_mode", type=str, default='mix')
args = parser.parse_args()
pad_i=0

random.seed(args.seed)
np.random.seed(args.seed)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

n_tags=['NN','NNS','NNP','NNPS']
adj_tags=['JJ','JJR','JJS']
adv_tags=['RB','RBR','RBS']
v_tags=['VB','VBD','VBG','VBN','VBP','VBZ']


def mix_aug(text_):
    text = deepcopy(text_)

    # num of aug tokens    
    aug_num=int(0.1*len(text)+1)
    # random choose index of aug tokens
    aug_index_list=list(range(len(text)))
    random.shuffle(aug_index_list)
    aug_index_list=aug_index_list[:aug_num]
    
    # i: token index in [text]
    for i in aug_index_list:
        aug_choice=random.randint(0,1)
        # UNK mask
        if(aug_choice==0):
            text[i]='[UNK]'

        # SYN replacement
        elif(aug_choice==1):
            synonyms=wordnet.synsets(text[i])
            # UNK for bert subwords
            if len(synonyms) == 0:
                text[i] == '[UNK]'
            # to token lists
            ori_choices=sorted(list(set(chain.from_iterable([word.lemma_names() for word in synonyms]))))

            for syn in ori_choices:
                if syn not in tokenizer.vocab:
                    ori_choices.remove(syn)
            if( text[i] in ori_choices):
                ori_choices.remove(text[i])
            if(len(ori_choices)==0):
                continue
            random.shuffle(ori_choices)
            replace_word=ori_choices[0]
            
            text[i]=replace_word
        
        assert len(text_) == len(text)
    return text


# ulb_file = open(os.path.join("./data/", args.ds, 'aug_unlabel_tar_train.tsv'), 'r')
ulb_file = open(os.path.join("./data/", args.ds, 'unlabel_tar_train.tsv'), 'r')
ulb_file_lines = ulb_file.readlines()[1:]

rows = []
for i in trange(len(ulb_file_lines)):
    line = ulb_file_lines[i]
    # s_id, sentence, _ = line.strip().split('\t')
    s_id, sentence, senti = line.strip().split('\t')

    # sentence => tokens => insert sep => subword tokenization => augmentation 
    sentence = sentence.split() # to tokens

    # insert cls 
    sentence.insert(0, '[CLS]')
    sentence.append('[SEP]')

    # bert tokenize first, avoid || be masked
    tokens = [] 

    for i, word in enumerate(sentence):
        token = tokenizer.tokenize(word)
        tokens.extend(token)
    
    sentence = tokens 

    # if args.aug_mode == 'mix':
    #     aug_text = mix_aug(sentence)


    sentence = ' '.join(sentence)
    # aug = ' '.join(aug_text)
    # row = [s_id.strip(), sentence.strip(), aug.strip()]
    row = [s_id.strip(), sentence.strip(), senti]
    rows.append(row)



with open(os.path.join('./data',args.ds,'bert_tar_unlabel_train_'+str(args.cur_run_times)+'.tsv'),'w', encoding='utf-8') as w:
    w.write('s_id\tsentence\tsenti\n')
    
    tsv_w = csv.writer(w, delimiter='\t',quoting=csv.QUOTE_NONE, quotechar=None)
    tsv_w.writerows(rows)
    
