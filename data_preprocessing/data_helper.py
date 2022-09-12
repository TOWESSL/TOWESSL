import codecs
import torch
import gensim
import pickle
import numpy as np
from tqdm import tqdm,trange

def load_text_target_label(path,allow_repeat=True):
    text_list = []
    target_list = []
    label_list = []
    id_list=[]
    with open(path) as fo:
        lines=fo.readlines()
        for line_index in trange(len(lines)):
            if(line_index==0):continue
            line=lines[line_index]
            s_id, sentence, target_tags, opinion_words_tags = line.split('\t')
            #s_id, sentence, target_tags, opinion_words_tags,pos_tags = line.split('\t')
            if(sentence.strip() in text_list and allow_repeat==False):continue
            id_list.append(s_id.strip())
            text_list.append(sentence.strip())
            w_t = target_tags.strip().split(' ')
            target = [t.split('\\')[-1] for t in w_t]
            target_list.append(target)
            w_l = opinion_words_tags.strip().split(' ')
            label = [l.split('\\')[-1] for l in w_l]
            label_list.append(label)
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
            if((i+1)%10000==0):
                print('loading %d'%(i+1))

    return id_list,text_list, target_list, label_list#,pos_tag_list
def load_text_target(path,allow_repeat=True):
    text_list = []
    target_list = []
    id_list=[]
    with open(path) as fo:
        lines=fo.readlines()
        for line_index in trange(len(lines)):
            if(line_index==0):continue
            line=lines[line_index]
            s_id, sentence, target_tags = line.split('\t')
            if(sentence.strip() in text_list and allow_repeat==False):continue
            id_list.append(s_id.strip())
            text_list.append(sentence.strip())
            w_t = target_tags.strip().split(' ')
            target = [t.split('\\')[-1] for t in w_t]
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
        for line_index in trange(len(lines),desc='loading data'):
            if(line_index==0):continue
            line=lines[line_index]
            s_id, sentence = line.split('\t')
            if(sentence.strip() in text_list and allow_repeat==False):continue
            id_list.append(s_id.strip())
            text_list.append(sentence.strip())

    return id_list,text_list


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


