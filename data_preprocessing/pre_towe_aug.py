from data_helper import load_text_target_label,load_text_target,load_text
import codecs
import nltk
from itertools import chain
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
from nltk.corpus import wordnet
from tqdm import tqdm,trange
import os
import random
import pickle
import math
import copy
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
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
parser.add_argument("--uda_p", type=int, default=0.7)
parser.add_argument("--seed", type=int, default=100)
parser.add_argument("--voc_name", type=str, default='vocabulary.pkl')
parser.add_argument("--embed_name", type=str, default='embedding_table.npy')
parser.add_argument("--aug_mode", type=str, default='mix')
args = parser.parse_args()
pad_i=0
np.random.seed(args.seed)
random.seed(args.seed)
#_,train_text, train_target= load_text_target(os.path.join("./data/", args.ds, 'tar_train.tsv'))
#_,u_train_text=load_text(os.path.join("./data/", args.ds, 'unlabel_tar_train.tsv'))
word2index = pickle.load(open(os.path.join('./common_data/embedding', args.voc_name), "rb"))
init_embedding = np.load(os.path.join('./common_data/embedding', args.embed_name))
init_embedding = np.float32(init_embedding)
index2word = {}
n_tags=['NN','NNS','NNP','NNPS']
adj_tags=['JJ','JJR','JJS']
adv_tags=['RB','RBR','RBS']
v_tags=['VB','VBD','VBG','VBN','VBP','VBZ']


for key, value in word2index.items():
    index2word[value] = key
def create_tfidf_dic(u_train_text,uda_p):
    total_text=[]
    total_text.extend(u_train_text)
    total_text=set(total_text)
    corpus=list(total_text)
    corpus_num=len(corpus)
    print('corpus size',len(corpus))
    vectorizer = CountVectorizer(token_pattern=r'\b\w+\b')
    X = vectorizer.fit_transform(corpus)
    word = vectorizer.get_feature_names()
    transformer = TfidfTransformer()
    tfidf = transformer.fit_transform(X).toarray()
    X2=(X.toarray().sum(axis=0))
    X2=np.squeeze(X2)
    X3=np.where(X.toarray()>=1,1,0)
    X3=(X3.sum(axis=0))
    X3=np.squeeze(X3)
    word_prob=[]
    word_prob_dic={}
    Z2=0
    S=[]
    print('calculate word prob')
    for i in range(len(word)):
        S.append(X2[i]*math.log10(corpus_num/X3[i]))
    maxS=max(S)
    for i in range(len(word)):
        Z2+=maxS-S[i]
    for i in trange(len(word)):
        word_prob.append((maxS-S[i])/Z2)
        word_prob_dic[word[i]]=(maxS-S[i])/Z2
    word_prob_dic = sorted(word_prob_dic.items(), key=lambda x: x[1], reverse=True)
    #print(word_prob_dic)
    print('calculate sentence prob')
    sentence_prob={}
    for i in trange(len(corpus)):
        s=corpus[i].split(' ')
        tmp=[]
        for w in s:
            if(w in word):
                tmp.append(tfidf[i][word.index(w)])
            else:
                tmp.append(1)
        C=max(tmp)
        Z1=0
        for j in tmp:
            Z1+=(C-j)/len(tmp)
        for j in range(len(tmp)):
            if(Z1==0):break
            tmp[j]=min(1,uda_p*(C-tmp[j])/Z1)
        sentence_prob[corpus[i]]=tmp

    return word,word_prob,sentence_prob
def tfidf_aug(text,word2index,index2word,word,word_prob,sentence_prob):
    prob=sentence_prob[text]
    text=text.split(' ')
    for i in range(len(prob)):
        if(prob[i]>np.random.rand()):
            r_word = np.random.choice(word, p = np.array(word_prob).ravel())
            text[i]=r_word
    return ' '.join(text)
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

def unk_aug(text,word2index,index2word,a=None):
    text=text.split(' ')
    #text_pos_tag=nltk.pos_tag(text)
    aug_num=int(0.1*len(text)+1)
    aug_index_list=list(range(len(text)))
    random.shuffle(aug_index_list)
    aug_index_list=aug_index_list[:aug_num]
    for i in aug_index_list:
        text[i]='[UNK]'
    text=' '.join(text)
    return text
def syn_aug(text,word2index,index2word,a=None):
    text=text.split(' ')
    #text_pos_tag=nltk.pos_tag(text)
    aug_num=int(0.1*len(text)+1)
    aug_index_list=list(range(len(text)))
    random.shuffle(aug_index_list)
    aug_index_list=aug_index_list[:aug_num]
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
    text=' '.join(text)
    return text
def mix_aug(text,word2index,index2word,a=None):
    text=text.split(' ')
    #text_pos_tag=nltk.pos_tag(text)
    aug_num=int(0.1*len(text)+1)
    aug_index_list=list(range(len(text)))
    random.shuffle(aug_index_list)
    aug_index_list=aug_index_list[:aug_num]
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
    text=' '.join(text)
    return text

def make_target(target,aug_text):
    target=copy.deepcopy(target).split(' ')
    aug_target=copy.deepcopy(aug_text).split(' ')
    for i in range(len(target)):
        t=target[i].split('\\')[-1]
        aug_target[i]=aug_target[i]+'\\'+t
    return ' '.join(aug_target)

u_train_f= open(os.path.join("./data/", args.ds, 'unlabel_train_'+str(args.cur_run_times)+'.tsv'),'r')
u_train_f_lines=u_train_f.readlines()
tfidf_lines=[]
for i in range(len(u_train_f_lines)):
    tfidf_lines.append(u_train_f_lines[i].strip().split('\t')[1])
if(args.aug_mode=='tfidf'):
    word,word_prob,sentence_prob=create_tfidf_dic(tfidf_lines,args.uda_p)
f=open(os.path.join('./data',args.ds,'aug_unlabel_train_'+str(args.cur_run_times)+'.tsv'),'w')
f.write("s_id\ttarget\taug_target\tsentiment\n")
aug_size=0

for i in trange(len(u_train_f_lines),desc='write aug'):
    # u_train_senti=u_train_f_lines[i].strip().split('\t')[3]
    u_train_text=u_train_f_lines[i].strip().split('\t')[1]
    if(i==0):continue
    if(args.aug_mode=='unk'):
        aug_text=unk_aug(u_train_text,word2index,index2word)
    elif(args.aug_mode=='syn'):
        aug_text=syn_aug(u_train_text,word2index,index2word)
    elif(args.aug_mode=='mix'):
        aug_text=mix_aug(u_train_text,word2index,index2word)
    elif(args.aug_mode=='tfidf'):
        aug_text=tfidf_aug(u_train_text,word2index,index2word,word,word_prob,sentence_prob)
    if(aug_text==u_train_text):continue
    u_train_target=u_train_f_lines[i].strip().split('\t')[2]
    aug_target=make_target(u_train_target,aug_text)
    f.write('semi\t'+u_train_text+'\t'+u_train_target+'\t'+aug_text+'\t'+aug_target+'\n')
    aug_size+=1
f.close()
u_train_f.close()
print('aug size: ',aug_size)
