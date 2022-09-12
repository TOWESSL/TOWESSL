import codecs
from operator import delitem
from typing import Sequence, Text
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

import torch.nn as nn
from torch import optim

pad_i = 0
tag2id = {'B': 1, 'I': 2, 'O': 3 ,'X': 4, 'SEP':5, 'CLS': 6}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
def load_text_target_label(path,save_path, allow_repeat=True):
    
    text_list = []
    target_list = []
    label_list = []
    id_list=[]
    
    rows = []
    # tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    with open(path) as fo:
        lines=fo.readlines()
        lines=lines[1:]
    
        for line_index in trange(len(lines)):
            line=lines[line_index]
            s_id, sentence, target_tags, opinion_words_tags = line.split('\t')
            # Remove repeat sentence
            if(sentence.strip() in text_list and allow_repeat==False):continue

            # ID
            id_list.append(s_id.strip())

            # Sentence
            sentence=sentence.strip().split(' ')
            w_t = target_tags.strip().split(' ')
            target = [t.split('\\')[-1] for t in w_t]
            sentence.insert(0,'[CLS]')
            sentence.append('[SEP]')
            target.insert(0,'CLS')
            target.append('SEP')
            w_l = opinion_words_tags.strip().split(' ')
            label = [l.split('\\')[-1] for l in w_l]
            label.insert(0,'CLS')
            label.append('SEP')
            # insert ||        
            sentence, target, label = insert_seq(sentence, target, label)
              
            
            # tag X for subword ##
            sentence, target, label = subword_tag(sentence, target, label, tokenizer)

            sentence=' '.join(sentence)
            text_list.append(sentence.strip())
            target = ' '.join(target)
            target_list.append(target.strip())
            label = ' '.join(label)
            label_list.append(label.strip())

            row = [s_id.strip(), sentence.strip(), target.strip(), label.strip()]
            rows.append(row)

    with open(save_path, 'w', encoding='utf-8') as f:
        tsv_w = csv.writer(f, delimiter='\t',quoting=csv.QUOTE_NONE, quotechar=None)
        tsv_w.writerows(rows)

def numericalize_label(labels, vocab):
    label_tensor = []
    for i, label in enumerate(labels):
        label_tensor.append(vocab[label])
    return label_tensor

# Prepare distence dependecy
def numericalize_dep_dis_list(text_list,label_list):
    dic={}
    for i in trange(len(text_list),desc='making dependency distance'):
        key=text_list[i]+' '.join(label_list[i])
        val=numericalize_dep_dis(text_list[i],label_list[i])
        dic[key]=val
    return dic

def numericalize_dep_dis(text,labels):
    text=' '.join(text.split()[1:-1]).replace('#', '')
    doc = nlp(text)
    edges = []
    for token in doc:
        edges.append(('{0}-{1}'.format(token.text,token.i),'{0}-{1}'.format(token.text,token.i)))
        for child in token.children:
             edges.append(('{0}-{1}'.format(token.text,token.i),'{0}-{1}'.format(child.text,child.i)))
    graph = nx.Graph(edges)
    tar=np.asarray(labels[1:-1])
    tar1=np.where(tar=='I',1,0)
    tar2=np.where(tar=='B',1,0)
    tar = tar1 | tar2
    tar_words=[]
    text=text.split()
    for i in range(len(tar)):
        if(tar[i]==1):
            tar_words.append('{0}-{1}'.format(text[i],i))
    dep_dis_list=[]
    for tar_word in tar_words:
        dep_dis=[]
        for w in range(len(text)):
            try:
                tmp=nx.shortest_path_length(graph, source='{0}-{1}'.format(text[w],w), target=tar_word)
                dep_dis.append(tmp)
            except nx.NetworkXNoPath:
                dep_dis.append(200)
        dep_dis_list.append(dep_dis)
    res=np.asarray(dep_dis_list)
    dis_pad=np.zeros_like(res[0])
    for i in range(res.shape[0]): 
        dis_pad=np.where(res[i]==200,1,dis_pad)
    res=res.mean(axis=0)
    res=np.where(dis_pad==1,200,res)
    res=res.tolist()
    res.insert(0,200)
    res.append(200)
    return res

# Prepare Bert ids, attention mask, segment ids
def numericalize_bert(raw_text_list):
    text_list=copy.deepcopy(raw_text_list)
    for i in range(len(text_list)):
        text_list[i]=text_list[i].split()
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
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

# insert || for target-specific data preprocessing
def insert_seq(sentence, target, label):
    i = 0
    while i < len(sentence):
        cur_tag = target[i]
        if cur_tag == 'I' and target[i-1] != 'I' and target[i-1] != 'B':
            target[i] = 'B'
            continue
        if cur_tag == 'B':
            sentence.insert(i, '‖')
            target.insert(i, 'SEP')
            label.insert(i, 'SEP')
            i += 1
            if i == len(sentence) - 1:
                sentence.insert(i+1, '‖')
                target.insert(i+1, 'SEP')
                label.insert(i+1, 'SEP')
            elif target[i+1] != 'I':
                sentence.insert(i+1, '‖')
                target.insert(i+1, 'SEP')
                label.insert(i+1, 'SEP')
        if cur_tag == 'I':
            if i == len(sentence) - 1:
                sentence.insert(i+1, '‖')
                target.insert(i+1, 'SEP')
                label.insert(i+1, 'SEP')
            elif target[i+1] != 'I':
                sentence.insert(i+1, '‖')
                target.insert(i+1, 'SEP')
                label.insert(i+1, 'SEP')
        i+=1
    return sentence, target, label

# insert || for target-specific data preprocessing
def insert_seq_ulb(sentence, target):
    i = 0
    while i < len(sentence):
        cur_tag = target[i]
        if cur_tag == 'I' and target[i-1] != 'I' and target[i-1] != 'B':
            target[i] = 'B'
            continue
        if cur_tag == 'B':
            sentence.insert(i, '‖')
            target.insert(i, 'SEP')
            i += 1
            if i == len(sentence) - 1:
                sentence.insert(i+1, '‖')
                target.insert(i+1, 'SEP')
            elif target[i+1] != 'I':
                sentence.insert(i+1, '‖')
                target.insert(i+1, 'SEP')
        if cur_tag == 'I':
            if i == len(sentence) - 1:
                sentence.insert(i+1, '‖')
                target.insert(i+1, 'SEP')
            elif target[i+1] != 'I':
                sentence.insert(i+1, '‖')
                target.insert(i+1, 'SEP')
        i+=1
    assert len(sentence) == len(target)
    return sentence, target


# insert X tag for bert wordpiece
def subword_tag(sentence, target, label, tokenizer):

    tokens = []
    tags = []
    l_tags = []

    for i, word in enumerate(sentence):
        token = tokenizer.tokenize(word)
        tag1 = target[i]
        l_tag1 = label[i]
        tokens.extend(token)
        for m in range(len(token)):
            if m == 0:
                tags.append(tag1)
                l_tags.append(l_tag1)
            else:
                tags.append('X')
                l_tags.append('X')
    assert len(tokens) == len(tags) 
    assert len(tags) == len(l_tags)
    return tokens, tags, l_tags

# insert X tag for bert wordpiece
def subword_tag_ulb(sentence, target, tokenizer):
    assert len(sentence) == len(target)

    tokens = []
    tags = []

    for i, word in enumerate(sentence):
        token = tokenizer.tokenize(word)
        tag1 = target[i]
        ts = []
        for m in range(len(token)):
            if m == 0:
                ts.append(tag1)
            else:
                ts.append('X')
        assert len(token) == len(ts)
        tokens.extend(token)
        tags.extend(ts)

    assert len(tokens) == len(tags) 
    return tokens, tags


# load saved preprocessed labeled data
def load_labeled_set(path):

    ids = []
    sents = []
    targets = []
    labels = []

    with open(path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            id, sent, tar, label = line.strip().split('\t')
            ids.append(id)
            sents.append(sent)
            targets.append(tar.split())
            labels.append(label.split())
            try:
                assert len(sent.split()) == len(tar.split())
                assert len(label.split()) == len(tar.split())
            except:
                print(sent)
                print(tar)
                print(label)
    return ids, sents, targets, labels 

# Create Labeled Iterator
def create_labeled_dataset(texts, targets, labels, args, train=True):
    
    # Define Fileds
    BERT_IDS=data.Field(sequential=True, use_vocab=False, pad_token=0, batch_first=True)
    BERT_MASK=data.Field(sequential=True, use_vocab=False, pad_token=0, batch_first=True)
    BERT_SEG=data.Field(sequential=True, use_vocab=False, pad_token=0, batch_first=True)
    TEXT = data.Field(sequential=True, use_vocab=False, pad_token=0, batch_first=True, include_lengths=True)
    LABEL_T = data.Field(sequential=True, use_vocab=False,pad_token=pad_i, batch_first=True)
    LABEL_O = data.Field(sequential=True, use_vocab=False,pad_token=pad_i,  batch_first=True)
    DIS_T = data.Field(sequential=True, use_vocab=False,pad_token=200,  batch_first=True)
    TAR_OTH=data.Field(sequential=True, use_vocab=False,pad_token=pad_i,  batch_first=True)
    DEP_DIS_T = data.Field(sequential=True, use_vocab=False,pad_token=200,  batch_first=True)
    
    fields = [('bert_ids',BERT_IDS),('bert_mask',BERT_MASK),('bert_seg', BERT_SEG), ('target', LABEL_T),('dis',DIS_T),('label', LABEL_O)]

    # Split dev set
    if train:
        train_texts, train_t, train_ow, dev_texts, dev_t, dev_ow = split_dev(texts, targets, labels)
        dev_bert_ids_dic, dev_bert_mask_dic, dev_bert_seg_dic = numericalize_bert(dev_texts)
        # dev_dep_dis_dic=numericalize_dep_dis_list(dev_texts,dev_t)

        dev_data =  [[dev_bert_ids_dic[text],
                        dev_bert_mask_dic[text],
                        dev_bert_seg_dic[text],
                        numericalize_label(target,tag2id),
                        numericalize_dis(target, tag2id),
                        numericalize_label(label, tag2id),
                        ] for text, target, label in zip(dev_texts, dev_t, dev_ow)]
        dev_dataset = ToweDataset(fields, dev_data,'dev')
        dev_iter = data.Iterator(dev_dataset, batch_size=args.eval_bs, shuffle=True,
                                  repeat=False,
                                  device=device)
    else:
        train_texts = texts
        train_t = targets 
        train_ow = labels

    train_bert_ids_dic, train_bert_mask_dic, train_bert_seg_dic = numericalize_bert(train_texts)
    # train_dep_dis_dic=numericalize_dep_dis_list(train_texts,train_t)

    train_data = [[train_bert_ids_dic[text],
                train_bert_mask_dic[text],
                train_bert_seg_dic[text],
                numericalize_label(target,tag2id),
                numericalize_dis(target, tag2id),
                numericalize_label(label, tag2id),
                ] for text, target, label in zip(train_texts, train_t, train_ow)]
    
    if train:
        train_dataset = ToweDataset(fields, train_data,'train')
        train_iter = data.Iterator(train_dataset, batch_size=args.batch_size, repeat=False, shuffle=True,
                                   device=device)
        return train_iter, dev_iter
    else:
        train_dataset = ToweDataset(fields, train_data,'test')
        test_iter = data.Iterator(train_dataset, batch_size=args.eval_bs, shuffle=True,
                                  repeat=False,
                                  device=device)
        return test_iter

# split dev set
def split_dev(train_texts, train_t, train_ow):
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
    
    dev_texts, dev_t, dev_ow= ([train_texts[i] for i in dev_i_index], [train_t[i] for i in dev_i_index],
                                [train_ow[i] for i in dev_i_index])
    train_texts, train_t, train_ow = ([train_texts[i] for i in train_i_index], [train_t[i] for i in train_i_index],
                                        [train_ow[i] for i in train_i_index])
    
    return train_texts, train_t, train_ow, dev_texts, dev_t, dev_ow

def numericalize_dis(labels, vocab):
    label_tensor = []
    for i, label in enumerate(labels):
        label_tensor.append(vocab[label])
    label_tensor=np.array(label_tensor)
    distance=np.zeros_like(label_tensor)
    zero=np.zeros_like(label_tensor)
    label_tensor1=np.where(label_tensor == tag2id['B'], label_tensor, zero)
    label_tensor2=np.where(label_tensor == tag2id['I'], label_tensor, zero)
    label_tensor = label_tensor1 | label_tensor2
    mark=np.nonzero(label_tensor)
    start_list=[]
    end_list=[]
    start_list.append(mark[0][0])
    end_list.append(mark[0][-1])
    end_list.reverse()

    unmark=np.arange(distance.shape[0])
    s=unmark[start_list[0]]
    e=unmark[end_list[0]]
    unmark[:s]-=s
    unmark[s:e]=0
    unmark[e:]-=e
    distance=np.array(unmark,dtype=np.int64)
    distance=np.abs(distance)
    distance=distance.tolist()
    return distance

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

def load_aug_text_target(path,save_path,args, split=True,allow_repeat=True):
    text_list = []
    aug_list=[]
    target_list=[]
    aug_tar_list=[]
    id_list=[]

    rows = []
    
    with open(path) as fo:
        lines=fo.readlines()
        lines=lines[1:]

        file_len=len(lines)
        if(split==True):
            random.shuffle(lines)
            end_i=min(file_len,args.ulb_size)
        else:
            end_i=file_len
        
        for line_index in trange(end_i):
            line=lines[line_index]
            s_id, sentence,tar,aug,aug_tar, senti = line.strip().split('\t')
            if(sentence.strip() in text_list and allow_repeat==False):continue
            
            id_list.append(s_id.strip())
            
            sentence=sentence.split(' ')
            aug=aug.split(' ')
            tar=tar.split(' ')
            tar = [t.split('\\')[-1] for t in tar]
            aug_tar=aug_tar.strip().split(' ')
            aug_tar = [t.split('\\')[-1] for t in aug_tar]

            # insert ||
            sentence, tar = insert_seq_ulb(sentence, tar)
            aug, aug_tar = insert_seq_ulb(aug, aug_tar)

            assert len(sentence) == len(tar)
            assert len(sentence) == len(aug)
            # tag X for subword ##
            # sentence, tar = subword_tag_ulb(sentence, tar, tokenizer)
            # aug, aug_tar = subword_tag_ulb(aug, aug_tar, tokenizer)

            sentence=' '.join(sentence)
            text_list.append(sentence.strip())

            aug=' '.join(aug)
            aug_list.append(aug.strip())

            tar = ' '.join(tar)
            target_list.append(tar)
            
            aug_tar = ' '.join(aug_tar)
            aug_tar_list.append(aug_tar)

            
            row = [s_id.strip(), sentence.strip(), tar.strip(), aug.strip(), aug_tar.strip(), senti]
            rows.append(row)
    with open(save_path, 'w', encoding='utf-8') as f:
        tsv_w = csv.writer(f, delimiter='\t',quoting=csv.QUOTE_NONE, quotechar=None)
        tsv_w.writerows(rows)


def make_u_batch_iter(args):
    
    if not os.path.exists(os.path.join("../data/", args.ds, 'preprocessed_aug_unlabel_train_'+str(args.cur_run_times)+'_'+str(args.ulb_size)+'.tsv')):
        load_aug_text_target(os.path.join("../data/", args.ds, 'bert_aug_unlabel_train_'+str(args.cur_run_times)+'.tsv'), os.path.join("../data/", args.ds, 'preprocessed_aug_unlabel_train_'+str(args.cur_run_times)+'_'+str(args.ulb_size)+'.tsv'), args)
    
    _,u_train_text,u_train_target,aug_u_train_text,aug_u_train_target, sentis = load_ulb_data(os.path.join("../data/", args.ds, 'preprocessed_aug_unlabel_train_'+str(args.cur_run_times)+'_'+str(args.ulb_size)+'.tsv'))
    
    u_train_raw_data=(u_train_text,u_train_target,aug_u_train_text,aug_u_train_target, sentis)

    # FIELDS        
    BERT_IDS=data.Field(sequential=True, use_vocab=False, pad_token=0, batch_first=True)
    BERT_MASK=data.Field(sequential=True, use_vocab=False, pad_token=0, batch_first=True)
    BERT_SEG=data.Field(sequential=True, use_vocab=False, pad_token=0, batch_first=True)
    TEXT = data.Field(sequential=True, use_vocab=False, pad_token=0, batch_first=True, include_lengths=True)
    LABEL_T = data.Field(sequential=True, use_vocab=False,pad_token=pad_i, batch_first=True)
    LABEL_O = data.Field(sequential=True, use_vocab=False,pad_token=pad_i,  batch_first=True)
    DIS_T = data.Field(sequential=True, use_vocab=False,pad_token=200,  batch_first=True)
    TAR_OTH=data.Field(sequential=True, use_vocab=False,pad_token=pad_i,  batch_first=True)
    DEP_DIS_T = data.Field(sequential=True, use_vocab=False,pad_token=200,  batch_first=True) 
    SENTI = data.Field(sequential=False, use_vocab=False, batch_first=True)


    u_fields=[('bert_ids',BERT_IDS),('bert_mask',BERT_MASK),('bert_seg', BERT_SEG),
                ('aug_bert_ids',BERT_IDS),('aug_bert_mask',BERT_MASK),('aug_bert_seg', BERT_SEG),
                ('target', LABEL_T),('dis',DIS_T),('senti', SENTI)
            ]
    
    u_train_bert_ids_dic, u_train_bert_mask_dic, u_train_bert_seg_dic = numericalize_bert(u_train_raw_data[0])
    aug_u_train_bert_ids_dic,aug_u_train_bert_mask_dic, aug_u_train_seg_dic = numericalize_bert(u_train_raw_data[2])
    
    u_train_data = [[u_train_bert_ids_dic[text],u_train_bert_mask_dic[text],u_train_bert_seg_dic[text],
                    aug_u_train_bert_ids_dic[aug_text],aug_u_train_bert_mask_dic[aug_text],aug_u_train_seg_dic[aug_text],
                    numericalize_label(target, tag2id),numericalize_dis(target, tag2id),
                    senti,
                    ] for text,target,aug_text,aug_target,senti in zip(*u_train_raw_data)]

    u_train_dataset = ToweDataset(u_fields, u_train_data,'u_train')
    u_train_iter = data.Iterator(u_train_dataset, batch_size=args.u_batch_size, repeat=False, shuffle=True,
                               device=device)
    #u_train_iter=make_senti_u_batch_iter(senti_model,u_train_iter,W,word2index,index2word,args)
    return u_train_dataset

# load saved preprocessed unlabeled data
def load_ulb_data(path):
    ids = []
    sents = []
    augs = []
    targets = []
    aug_tars = []
    sentis = []

    with open(path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            id, sent, tar, aug, aug_tar, senti = line.strip().split('\t')
            ids.append(id)
            sents.append(sent)
            targets.append(tar.split())
            augs.append(aug)
            aug_tars.append(aug_tar.split())
            sentis.append(senti)

    return ids, sents, targets, augs, aug_tars, sentis

def pretrain_senti_model(model, dataset, args):
    loss = nn.CrossEntropyLoss()
    train_iter = data.Iterator(dataset, batch_size=args.u_batch_size, repeat=False, shuffle=True,device=device)

    bert_params = list(map(id, model.bert_model.parameters()))
    base_params = filter(lambda p: id(p) not in bert_params,model.parameters())
    
    params = [{'params': base_params},
              {'params': model.bert_model.parameters(), 'lr': 1e-5},
              ]

    optimizer = optim.Adam(params, lr=1e-4)
    model.train()
    for i in trange(args.senti_iter):
        try:
            batch=next(iter(train_iter))
        except:
            train_iter = data.Iterator(dataset, batch_size=args.senti_batch_size, repeat=False, shuffle=True,device=device)
            batch = next(iter(train_iter))
        
        logits, _ = model(batch)

        l = loss(logits, batch.senti)
        l.backward()
        optimizer.step()
        model.zero_grad()
    
    # Eval
    try:
        batch=next(iter(train_iter))
    except:
        train_iter = data.Iterator(dataset, batch_size=args.senti_batch_size, repeat=False, shuffle=True,device=device)
        batch = next(iter(train_iter))
    total_num = 0
    correct_num = 0
    for i in trange(15):
        try:
            batch=next(iter(train_iter))
        except:
            train_iter = data.Iterator(dataset, batch_size=args.senti_batch_size, repeat=False, shuffle=True,device=device)
            batch = next(iter(train_iter))
        
        logits, _ = model(batch)
        prob = torch.softmax(logits, dim=-1)
        _ , pred = torch.max(prob, dim=-1)
        correct_num += (pred == batch.senti).sum()
        total_num += pred.size(0)
    print("Senti Eval Result: ", correct_num/total_num)

    torch.save(model.state_dict(), os.path.join('./backup', args.senti_model))

def evaluate_senti(model, dataset, args):
    loss = nn.CrossEntropyLoss()
    train_iter = data.Iterator(dataset, batch_size=args.u_batch_size, repeat=False, shuffle=True,device=device)

    bert_params = list(map(id, model.bert_model.parameters()))
    base_params = filter(lambda p: id(p) not in bert_params,model.parameters())
    
    params = [{'params': base_params},
              {'params': model.bert_model.parameters(), 'lr': 1e-5},
              ]

    optimizer = optim.Adam(params, lr=1e-4)
    
    model.eval()
    # Eval
    try:
        batch=next(iter(train_iter))
    except:
        train_iter = data.Iterator(dataset, batch_size=args.senti_batch_size, repeat=False, shuffle=True,device=device)
        batch = next(iter(train_iter))
    total_num = 0
    correct_num = 0
    for i in trange(300):
        try:
            batch=next(iter(train_iter))
        except:
            train_iter = data.Iterator(dataset, batch_size=args.senti_batch_size, repeat=False, shuffle=True,device=device)
            batch = next(iter(train_iter))
        
        logits, _ = model(batch)
        prob = torch.softmax(logits, dim=-1)
        _ , pred = torch.max(prob, dim=-1)
        correct_num += (pred == batch.senti).sum()
        total_num += pred.size(0)
    print("Senti Eval Result: ", correct_num/total_num)


