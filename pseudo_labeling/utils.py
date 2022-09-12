import codecs
from operator import delitem
from typing import Text
from nltk import tag, text
from numpy.core.fromnumeric import repeat
import torch
import gensim
import pickle
import random
import numpy as np
from tqdm import tqdm,trange
import copy
from transformers import BertTokenizer, BertModel
import nltk
from itertools import chain
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
from nltk.corpus import wordnet
import spacy
import networkx as nx
from spacy.tokens import Doc
from spacy.tokenizer import Tokenizer
nlp = spacy.load("en_core_web_sm")

import os 
import csv
import torchtext.data as data 

pad_i = 0
tag2id = {'B': 1, 'I': 2, 'O': 3 ,'X': 4, 'SEP':5, 'CLS': 6}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def load_text_target(path, save_path,  allow_repeat=True, tokenizer=tokenizer):
    """Load labeled set from path
        1. Load data
        2. Add [CLS] [SEP]
        3. Tokenize with BertTokenizer
        4. Alignment target

    Args:
        path ([str]): labeled data path
        save_path ([str]): save_path for preprocessed data
        
        allow_repeat (bool, optional): [description]. Defaults to True.
    """
    text_list = []
    target_list = []
    id_list=[]
    rows = []

    with open(path) as fo:
        lines=fo.readlines()
        lines=lines[1:]
    
        for line_index in trange(len(lines)):
            line=lines[line_index]
            s_id, sentence, target_tags = line.split('\t')
            # Remove repeat sentence
            if(sentence.strip() in text_list and allow_repeat==False):continue

            # ID
            id_list.append(s_id.strip())
            
            # Sentence
            sentence=sentence.strip().split(' ')
            w_t = target_tags.strip().split(' ')
            target = [t.split('\\')[-1] for t in w_t]
            # add [CLS] [SEP]
            sentence.insert(0,'[CLS]')
            sentence.append('[SEP]')
            target.insert(0,'CLS')
            target.append('SEP')
            # tag X with tokenizer
            sentence, target = subword_tag(sentence, target, tokenizer)

            sentence=' '.join(sentence)
            text_list.append(sentence.strip())
            target = ' '.join(target)
            target_list.append(target.strip())

            row = [s_id.strip(), sentence.strip(), target.strip()]
            rows.append(row)
    
    with open(save_path, 'w', encoding='utf-8') as f:
        tsv_w = csv.writer(f, delimiter='\t', quoting=csv.QUOTE_NONE, quotechar=None)
        tsv_w.writerows(rows)

def load_labeld_set(path):
    """Load preprocessed data from file

    Args:
        path ([type]): preprocessed_.tsv
    """
    ids = []
    sents = []
    targets = []

    with open(path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            id, sent, tar = line.strip().split('\t')
            ids.append(id)
            sents.append(sent)
            targets.append(tar.split())
    
    return ids, sents, targets

def subword_tag(sentence, target, tokenizer=tokenizer):
    """Tokenize given sentence and target with BertTokenizer

    Args:
        sentence (list): List of words of one sentence
        target (list): List of tags
        tokenizer (BertTokenizer, optional): BertTokenizer. Defaults to tokenizer.
    """
    assert len(sentence) == len(target)
    tokens = []
    tags = []

    for i, word in enumerate(sentence):
        token = tokenizer.tokenize(word)
        tag1 = target[i]
        tokens.extend(token)
        for m in range(len(token)):
            if m == 0:
                tags.append(tag1)
            else:
                tags.append('X')
    assert len(tokens) == len(tags)
    return tokens, tags

class ToweDataset(data.Dataset):

    @staticmethod
    def sort_key(ex):
        return len(ex.text)

    def __init__(self, fields, input_data,name, **kwargs):
        """Create an SemEval dataset instance given a text list and a label list.
        Arguments:
            text_field: The field that will be used for text data.
            label_field: The field that will be used for aspect data.
            input_data: a tuple contains texts and labels
            Remaining keyword arguments: Passed to the constructor of data.Dataset.
        """
        examples = []
        for e in input_data:
            examples.append(data.Example.fromlist(e, fields))
        print(name,' size ',len(examples))
        super(ToweDataset, self).__init__(examples, fields, **kwargs)

def numericalize(text, vocab):
    tokens = text.split()
    ids = []
    for token in tokens:
        if token in vocab:
            ids.append(vocab[token])
        else:
            ids.append(vocab['[UNK]'])
    assert len(ids) == len(tokens)
    return ids

def create_labeled_dataset(texts, targets, args, train=False):
    """create dataIter for labeled set

    Args:
        texts ([list of str]): texts
        targets ([double list]): taggers
        args ([argparser]): args
        train (bool, optional): Train or Test. Defaults to False.
    """
    # Define Fields
    BERT_IDS=data.Field(sequential=True, use_vocab=False, pad_token=0, batch_first=True)
    BERT_MASK=data.Field(sequential=True, use_vocab=False, pad_token=0, batch_first=True)
    BERT_SEG=data.Field(sequential=True, use_vocab=False, pad_token=0, batch_first=True)

    LABEL_T = data.Field(sequential=True, use_vocab=False,pad_token=pad_i, batch_first=True)

    fields = [('bert_ids',BERT_IDS),('bert_mask',BERT_MASK),('bert_seg', BERT_SEG), ('target', LABEL_T)]

    if train:
        # Split dev set
        train_texts, train_t, dev_texts, dev_t= split_dev(texts, targets)
        dev_bert_ids_dic, dev_bert_mask_dic, dev_bert_seg_dic = numericalize_bert(dev_texts)

        dev_data = [[dev_bert_ids_dic[text],
                    dev_bert_mask_dic[text],
                    dev_bert_seg_dic[text],
                    numericalize_label(target, tag2id)
                    ] for text, target in zip(dev_texts, dev_t)]
        dev_dataset = ToweDataset(fields, dev_data, 'dev')
        dev_iter = data.Iterator(dev_dataset,
                                batch_size = args.eval_bs,
                                shuffle=True,
                                repeat=False,
                                device=device
                                )
    else:
        train_texts = texts
        train_t = targets
    
    
    train_bert_ids_dic, train_bert_mask_dic, train_bert_seg_dic = numericalize_bert(train_texts)

    train_data = [[train_bert_ids_dic[text],
                train_bert_mask_dic[text],
                train_bert_seg_dic[text],
                numericalize_label(target, tag2id)
                ] for text, target in zip(train_texts, train_t)]
    
    if train:
        train_dataset = ToweDataset(fields, train_data, 'train')
        train_iter = data.Iterator(train_dataset,
                                    batch_size = args.batch_size, 
                                    repeat = False,
                                    shuffle=True, 
                                    device = device
                                    )
        return train_iter, dev_iter 
    else:
        train_dataset = ToweDataset(fields, train_data, 'test')
        test_iter = data.Iterator(train_dataset,
                                    batch_size = args.eval_bs, 
                                    repeat = False,
                                    shuffle=True, 
                                    device = device
                                    )
        return test_iter

def split_dev(train_texts, train_t):
    instances_index = []
    curr_s = ""
    curr_i = -1
    for i, s in enumerate(train_texts):
        s = ' '.join(s)

        if s == curr_s:
            instances_index[curr_i].append(i)
        else:
            curr_s = s
            instances_index.append([i])
            curr_i += 1
    print(curr_i)
    print(len(instances_index))
    assert curr_i+1 == len(instances_index)
    length = len(instances_index)
    index_list = np.random.permutation(length).tolist()
    train_index = [instances_index[i] for i in index_list[0: length - length//5]]  
    dev_index = [instances_index[i] for i in index_list[length - length//5:]]
    train_i_index = [i for l in train_index for i in l]
    dev_i_index = [i for l in dev_index for i in l]
    dev_texts, dev_t= ([train_texts[i] for i in dev_i_index], [train_t[i] for i in dev_i_index])
    train_texts, train_t = ([train_texts[i] for i in train_i_index], [train_t[i] for i in train_i_index])
    return train_texts, train_t, dev_texts, dev_t

def numericalize_bert(raw_text_list):
    """Prepare Bert ids, attention mask, segment ids

    Args:
        raw_text_list (list): input text
    """

    text_list=copy.deepcopy(raw_text_list)
    for i in range(len(text_list)):
        text_list[i]=text_list[i].split()
    
    ids_dic={}
    mask_dic={}
    seg_dic = {}
    for i in range(len(text_list)):
        ids = tokenizer.convert_tokens_to_ids(text_list[i])
        mask = [1]*len(ids)
        seg = [0] * len(ids)
        ids_dic[' '.join(text_list[i])]=ids
        mask_dic[' '.join(text_list[i])]=mask
        seg_dic[' '.join(text_list[i])]=seg
    return ids_dic,mask_dic,seg_dic

def numericalize_label(labels, vocab):
    label_tensor = []
    for i, label in enumerate(labels):
        label_tensor.append(vocab[label])
    return label_tensor

def make_u_batch_iter(args):
    u_train_text=[]
    aug_u_train_text=[]

    if not os.path.exists(os.path.join('../../data',args.ds,'preprocessed_tar_aug_unlabel_train_'+str(args.cur_run_times)+'.tsv')):
        load_aug_text(os.path.join('../../data',args.ds,'bert_tar_aug_unlabel_train_'+str(args.cur_run_times)+'.tsv'), os.path.join('../../data',args.ds,'preprocessed_tar_aug_unlabel_train_'+str(args.cur_run_times)+'.tsv'))

    _, u_train_text, aug_u_train_text = load_ulb_data(os.path.join('../../data',args.ds,'preprocessed_tar_aug_unlabel_train_'+str(args.cur_run_times)+'.tsv'))
    u_train_raw_data=(u_train_text, aug_u_train_text)


    # Fields
    BERT_IDS=data.Field(sequential=True, use_vocab=False, pad_token=0, batch_first=True)
    BERT_MASK=data.Field(sequential=True, use_vocab=False, pad_token=0, batch_first=True)
    BERT_SEG=data.Field(sequential=True, use_vocab=False, pad_token=0, batch_first=True)
    u_fields=[('bert_ids',BERT_IDS),('bert_mask',BERT_MASK),('bert_seg', BERT_SEG),
                ('aug_bert_ids',BERT_IDS),('aug_bert_mask',BERT_MASK),('aug_bert_seg', BERT_SEG),
            ]

    u_train_bert_ids_dic, u_train_bert_mask_dic, u_train_bert_seg_dic = numericalize_bert(u_train_raw_data[0])
    aug_u_train_bert_ids_dic,aug_u_train_bert_mask_dic, aug_u_train_seg_dic = numericalize_bert(u_train_raw_data[1])
    
    u_train_data = [[u_train_bert_ids_dic[text],u_train_bert_mask_dic[text],u_train_bert_seg_dic[text],
                    aug_u_train_bert_ids_dic[aug_text],aug_u_train_bert_mask_dic[aug_text],aug_u_train_seg_dic[aug_text],
                    ] for text,aug_text in zip(*u_train_raw_data)]

    u_train_dataset = ToweDataset(u_fields, u_train_data,'u_train')

    u_train_iter = data.Iterator(u_train_dataset, batch_size=args.u_batch_size, repeat=False, shuffle=True,
                               device=device)
    return u_train_dataset 
 
def load_aug_text(path, save_path,split=True, allow_repeat=True):
    text_list = []
    aug_list=[]
    id_list=[]
    rows = []

    with open(path) as fo:
        lines=fo.readlines()
        lines=lines[1:]
        
        file_len=len(lines)
        if split == True:
            random.shuffle(lines)
            end_i=min(file_len,100000)
        else:
            end_i = file_len

        for line_index in trange(end_i):
            line=lines[line_index]
            s_id, sentence, aug = line.split('\t')
            
            if(sentence.strip() in text_list and allow_repeat==False):continue
            
            id_list.append(s_id.strip())
            
            sentence=sentence.split(' ')
            aug = aug.split(' ')
            
            # sentence.insert(0,'[CLS]')
            # sentence.append('[SEP]')
            sentence=' '.join(sentence)
            text_list.append(sentence.strip())

            # aug.insert(0,'[CLS]')
            # aug.append('[SEP]')
            aug=' '.join(aug)
            aug_list.append(aug.strip())

            row = [s_id.strip(), sentence.strip(), aug.strip()]
            rows.append(row)
    
    with open(save_path, 'w', encoding='utf-8')as f:
        tsv_w = csv.writer(f, delimiter='\t',quoting=csv.QUOTE_NONE, quotechar=None)
        tsv_w.writerows(rows)
            

def load_ulb_data(path):
    ids = []
    sents = []
    sentis = []

    with open(path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            id, sent, senti = line.strip().split('\t')
            ids.append(id)
            sents.append(sent)
            sentis.append(senti)

    return ids, sents, sentis


def load_aug_text_all(path, save_path, allow_repeat=True, args=None):
    text_list = []
    senti_list=[]
    id_list=[]

    rows = []

    with open(path) as fo:
        lines=fo.readlines()
        lines=lines[1:]
        if args == None:
            file_len=len(lines)
        else:
            file_len = min(args.raw_size, len(lines))

        for line_index in trange(file_len):

            line=lines[line_index]
            s_id, sentence,senti = line.strip().split('\t')
            
            if(sentence.strip() in text_list and allow_repeat==False):continue
                
            id_list.append(s_id.strip())
            sentence=sentence.split(' ')
            sentence.insert(0,'[CLS]')
            sentence.append('[SEP]')
            sentence=' '.join(sentence)
            text_list.append(sentence.strip())
            
            senti_list.append(senti)

            row = [s_id.strip(), sentence.strip(), senti]
            # print(row)
            rows.append(row)

    with open(save_path, 'w', encoding='utf-8')as f:
        tsv_w = csv.writer(f, delimiter='\t',quoting=csv.QUOTE_NONE, quotechar=None)
        tsv_w.writerows(rows)


    # return id_list,text_list,aug_list



