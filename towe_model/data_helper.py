import codecs
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

class WhitespaceTokenizer(object):
    def __init__(self, vocab):
        self.vocab = vocab

    def __call__(self, text):
        words = text.split(' ')
        # All tokens 'own' a subsequent space character in this tokenizer
        spaces = [True] * len(words)
        return Doc(self.vocab, words=words, spaces=spaces)
nlp.tokenizer=WhitespaceTokenizer(nlp.vocab)
random.seed(1)
np.random.seed(1)
pad_i=0
tag2id = {'B': 1+1, 'I': 2+1, 'O': 0+1}
senti2id={'positive':0,'negtive':1}
id2senti={0:'positive',1:'negtive'}
taroth2id = {'B': 1+1, 'I': 1+1, 'O': 0+1}

def load_text_target_label(path,allow_repeat=True):
    text_list = []
    target_list = []
    label_list = []
    id_list=[]
    with open(path) as fo:
        lines=fo.readlines()
        lines=lines[1:]
        for line_index in trange(len(lines)):
            line=lines[line_index]
            s_id, sentence, target_tags, opinion_words_tags = line.split('\t')
            #s_id, sentence, target_tags, opinion_words_tags,pos_tags = line.split('\t')
            if(sentence.strip() in text_list and allow_repeat==False):continue
            id_list.append(s_id.strip())
            sentence=sentence.split(' ')
            sentence.insert(0,'[CLS]')
            sentence.append('[SEP]')
            sentence=' '.join(sentence)
            text_list.append(sentence.strip())
            w_t = target_tags.strip().split(' ')
            target = [t.split('\\')[-1] for t in w_t]
            target.insert(0,'O')
            target.append('O')
            target_list.append(target)
            w_l = opinion_words_tags.strip().split(' ')
            label = [l.split('\\')[-1] for l in w_l]
            label.insert(0,'O')
            label.append('O')
            label_list.append(label)

    return id_list,text_list, target_list, label_list

def load_text_target(path,allow_repeat=True):
    text_list = []
    target_list = []
    id_list=[]
    with open(path) as fo:
        lines=fo.readlines()
        lines=lines[1:]
        for line_index in trange(len(lines)):
            line=lines[line_index]
            s_id, sentence, target_tags = line.split('\t')
            if(sentence.strip() in text_list and allow_repeat==False):continue
            id_list.append(s_id.strip())
            sentence=sentence.split(' ')
            sentence.insert(0,'[CLS]')
            sentence.append('[SEP]')
            sentence=' '.join(sentence)
            text_list.append(sentence.strip())
            w_t = target_tags.strip().split(' ')
            target = [t.split('\\')[-1] for t in w_t]
            target.insert(0,'O')
            target.append('O')
            target_list.append(target)
            # w_p = pos_tags.strip().split(' ')
            # pos = [p.split('\\')[-1] for p in w_p]
            # pos_tag_list.append(pos)
            #print(text_list,target_list, label_list)
            #for t in range(len(target_list)):
            #    if '' in target_list[t]:
            #        print(text_list[t],target_list[t], label_list[t])
            #for t in range(len(label_list)):
            #    if '' in label_list[t]:
            #        print(text_list[t],target_list[t], label_list[t])

    return id_list,text_list, target_list

def load_text(path,allow_repeat=True):
    text_list = []
    id_list=[]
    with open(path) as fo:
        lines=fo.readlines()
        lines=lines[1:]
        for line_index in trange(len(lines)):
            line=lines[line_index]
            s_id, sentence = line.split('\t')
            if(sentence.strip() in text_list and allow_repeat==False):continue
            id_list.append(s_id.strip())
            sentence=sentence.split(' ')
            sentence.insert(0,'[CLS]')
            sentence.append('[SEP]')
            sentence=' '.join(sentence)
            text_list.append(sentence.strip())

    return id_list,text_list

def load_aug_text(path,allow_repeat=True):
    text_list = []
    aug_list=[]
    id_list=[]
    senti_list=[]
    with open(path) as fo:
        lines=fo.readlines()
        lines=lines[1:]
        file_len=len(lines)
        random.shuffle(lines)
        end_i=min(file_len,100000)
        for line_index in range(end_i):
            line=lines[line_index]
            s_id, sentence,aug,senti = line.split('\t')
            if(sentence.strip() in text_list and allow_repeat==False):continue
            id_list.append(s_id.strip())
            sentence=sentence.split(' ')
            sentence.insert(0,'[CLS]')
            sentence.append('[SEP]')
            sentence=' '.join(sentence)
            text_list.append(sentence.strip())
            aug=aug.split(' ')
            aug.insert(0,'[CLS]')
            aug.append('[SEP]')
            aug=' '.join(aug)
            aug_list.append(aug.strip())
            senti_list.append(senti.strip())

    return id_list,text_list,aug_list,senti_list


def load_aug_text_target(path,split=True,allow_repeat=True):
    text_list = []
    aug_list=[]
    target_list=[]
    aug_tar_list=[]
    id_list=[]
    with open(path) as fo:
        lines=fo.readlines()
        lines=lines[1:]
        file_len=len(lines)
        if(split==True):
            random.shuffle(lines)
            end_i=min(file_len,10000)
        else:
            end_i=file_len
        for line_index in range(end_i):
            line=lines[line_index]
            s_id, sentence,tar,aug,aug_tar = line.strip().split('\t')
            if(sentence.strip() in text_list and allow_repeat==False):continue
            id_list.append(s_id.strip())
            sentence=sentence.split(' ')
            sentence.insert(0,'[CLS]')
            sentence.append('[SEP]')
            sentence=' '.join(sentence)
            text_list.append(sentence.strip())
            aug=aug.split(' ')
            aug.insert(0,'[CLS]')
            aug.append('[SEP]')
            aug=' '.join(aug)
            aug_list.append(aug.strip())
            tar=tar.split(' ')
            aug_tar=aug_tar.strip().split(' ')
            tar = [t.split('\\')[-1] for t in tar]
            tar.insert(0,'O')
            tar.append('O')
            target_list.append(tar)
            aug_tar = [t.split('\\')[-1] for t in aug_tar]
            aug_tar.insert(0,'O')
            aug_tar.append('O')
            aug_tar_list.append(aug_tar)
    return id_list,text_list,target_list,aug_list,aug_tar_list

def generate_sentence_label(train_texts, train_ow):  # combine all ow labels for one sentence
    train_s_texts = []
    train_s_ow = []
    prev_text = ''
    train_s_t = []
    for i in range(len(train_texts)):
        if train_texts[i] != prev_text:
            prev_text = train_texts[i]
            train_s_texts.append(train_texts[i])
            train_s_ow.append([train_ow[i]])
        else:
            train_s_ow[-1].append(train_ow[i])
    print(len(train_s_texts))
    new_s_ow = []
    for t, o in zip(train_s_texts, train_s_ow):
        train_s_t.append([0 for i in range(len(t))])
        oarray = np.asarray(o)
        new_ow = oarray.max(axis=0).tolist()
        new_s_ow.append(new_ow)
        # print(str(t)+'\t'+str(o) + '\t' + str(new_ow))
    return train_s_texts, new_s_ow, new_s_ow


def numericalize_bert(raw_text_list):
    text_list=copy.deepcopy(raw_text_list)
    for text_i in range(len(text_list)):
        text_list[text_i]=text_list[text_i].split()
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    ids_dic={}
    mask_dic={}
    for text_i in range(len(text_list)):
        ids=tokenizer.convert_tokens_to_ids(text_list[text_i])
        mask=[1]*len(ids)
        ids_dic[' '.join(text_list[text_i])]=ids
        mask_dic[' '.join(text_list[text_i])]=mask
    return ids_dic,mask_dic


def numericalize(text, vocab):
    tokens= text.split()
    ids = []
    for token in tokens:
        if token in vocab:
            ids.append(vocab[token])
        else:
            ids.append(vocab['[UNK]'])
    assert len(ids) == len(tokens)
    return ids

def numericalize_senti(senti):
    if(senti=='positive'):
        return senti2id['positive']
    else:
        return senti2id['negtive']

def numericalize_label(labels, vocab):
    label_tensor = []
    for i, label in enumerate(labels):
        label_tensor.append(vocab[label])
    return label_tensor

def numericalize_dep_dis_list(text_list,label_list):
    dic={}
    for i in trange(len(text_list),desc='making dependency distance'):
        key=text_list[i]+' '.join(label_list[i])
        val=numericalize_dep_dis(text_list[i],label_list[i])
        dic[key]=val
    return dic

def numericalize_dep_dis(text,labels):
    text=' '.join(text.split()[1:-1])
    doc = nlp(text)
    edges = []
    for token in doc:
        edges.append(('{0}-{1}'.format(token.text,token.i),'{0}-{1}'.format(token.text,token.i)))
        for child in token.children:
             edges.append(('{0}-{1}'.format(token.text,token.i),'{0}-{1}'.format(child.text,child.i)))
    graph = nx.Graph(edges)
    tar=np.asarray(labels[1:-1])
    tar=np.where(tar!='O',1,0)
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
def numericalize_dis(labels, vocab):
    label_tensor = []
    for i, label in enumerate(labels):
        #print(labels,i,label)
        label_tensor.append(vocab[label])
    label_tensor=np.array(label_tensor)
    distance=np.zeros_like(label_tensor)
    zero=np.zeros_like(label_tensor)
    label_tensor=np.where(label_tensor >tag2id['O'], label_tensor, zero)
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
    
def numericalize_tar_oth(labels, vocab):
    #print(labels,vocab)
    label_tensor = []
    for i, label in enumerate(labels):
        #print(labels,i,label)
        label_tensor.append(vocab[label])
    return label_tensor

def translate_text(text,index2word):
    res=[]
    for i in range(text.shape[0]):
        tmp=[]
        for j in range(text.shape[1]):
            tmp.append(index2word[text[i,j].item()])
        res.append(tmp)
    return res
def find_tag(tag):
    res=''
    if(tag in n_tags):
        res='NOUN'
    elif(tag in v_tags):
        res='VERB'
    elif(tag in adj_tags):
        res='ADJ'
    elif(tag in adv_tags):
        res='ADV'
    return res
def create_sysets(word,tag):
    if(tag==''):
        return wordnet.synsets(word)
    elif (tag=='NOUN'):
        return wordnet.synsets(word,pos=wordnet.NOUN)
    elif(tag=='VERB'):
        return wordnet.synsets(word,pos=wordnet.VERB)
    elif(tag=='ADJ'):
        return wordnet.synsets(word,pos=wordnet.ADJ)
    elif(tag=='ADV'):
        return wordnet.synsets(word,pos=wordnet.ADV)

def unk_aug(prob,text,word2index,index2word,a=None):
    #text_pos_tag=nltk.pos_tag(text)
    text=copy.deepcopy(text)
    aug_num=int(0.1*len(text)+1)
    aug_index_list=np.random.choice(list(range(len(text))), size=aug_num, replace=False, p=prob).tolist()
    for i in aug_index_list:
        aug_choice=0
        text[i]='[UNK]'
    return text
def syn_aug(prob,text,word2index,index2word,a=None):
    #text_pos_tag=nltk.pos_tag(text)
    text=copy.deepcopy(text)
    aug_num=int(0.1*len(text)+1)
    aug_index_list=np.random.choice(list(range(len(text))), size=aug_num, replace=False, p=prob).tolist()
    for i in aug_index_list:
        #tag=find_tag(text_pos_tag[i][1])
        synonyms=wordnet.synsets(text[i])
        ori_choices=sorted(list(set(chain.from_iterable([word.lemma_names() for word in synonyms]))))
        choices=[]
        for j in range(len(ori_choices)):
            ori_choices[j]=ori_choices[j]
            if(ori_choices[j] in word2index):
                choices.append(ori_choices[j])
        if( text[i] in choices):
            choices.remove(text[i])
        if(len(choices)==0):
            continue
        random.shuffle(choices)
        replace_word=choices[0]
        if(replace_word not in word2index):
            print(replace_word)
        text[i]=replace_word
    return text

def mix_aug(prob,text,word2index,index2word,a=None):
    #text_pos_tag=nltk.pos_tag(text)
    text=copy.deepcopy(text)
    aug_num=int(0.1*len(text)+1)
    aug_index_list=np.random.choice(list(range(len(text))), size=aug_num, replace=False, p=prob).tolist()
    for i in aug_index_list:
        aug_choice=random.randint(0,1)
        if(aug_choice==0):
            text[i]='[UNK]'
        elif(aug_choice==1):
            #tag=find_tag(text_pos_tag[i][1])
            synonyms=wordnet.synsets(text[i])
            ori_choices=sorted(list(set(chain.from_iterable([word.lemma_names() for word in synonyms]))))
            choices=[]
            for j in range(len(ori_choices)):
                ori_choices[j]=ori_choices[j]
                if(ori_choices[j] in word2index):
                    choices.append(ori_choices[j])
            if( text[i] in choices):
                choices.remove(text[i])
            if(len(choices)==0):
                continue
            random.shuffle(choices)
            replace_word=choices[0]
            if(replace_word not in word2index):
                print(replace_word)
            text[i]=replace_word
    return text

