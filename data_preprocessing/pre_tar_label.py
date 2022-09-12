import codecs
import os
import pickle
import numpy as np
import copy
from tqdm import tqdm,trange
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
parser.add_argument("--origin", type=str, default='train.tsv')
parser.add_argument("--dest", type=str, default='tar_train.tsv')
parser.add_argument("--origin2", type=str, default='test.tsv')
parser.add_argument("--dest2", type=str, default='tar_test.tsv')
args = parser.parse_args()

def combine_tar(tags1,tags2):
    #print(tags1,tags2)
    tags1=tags1.split(' ')
    tags2=tags2.split(' ')
    res=copy.deepcopy(tags1)
    for i in range(len(res)):
        if(res[i]!=tags2[i]):
            if('\\O' in res[i]):
                res[i]=tags2[i]
            if('\\O' not in tags2[i] and '\\O' not in tags1[i] ):
                #print('error',tags1[i],tags2[i])
                if('\\I' in tags2[i]):res[i]=tags2[i]
                #print(' '.join(tags1))
                #print(' '.join(tags2))
                #print(' '.join(res))
    return ' '.join(res)

sentence_dic={}
with open(os.path.join('./data',args.ds,args.origin),'r') as f:
    lines=f.readlines()
    for line_index in trange(len(lines),desc="combine targets"):
        if(line_index==0):continue
        line=lines[line_index]
        s_id, sentence, target_tags, opinion_words_tags = line.split('\t')
        if(sentence not in sentence_dic):
            sentence_dic[sentence]=[s_id,target_tags]
        else:
            sentence_dic[sentence][1]=combine_tar(sentence_dic[sentence][1],target_tags)

with open(os.path.join('./data', args.ds, args.dest), 'w') as f:
    f.write("s_id\tsentence\ttarget_tags\n")
    for sentence in sentence_dic:
        f.write(sentence_dic[sentence][0]+'\t'+sentence+'\t'+sentence_dic[sentence][1]+'\n')

        
sentence_dic={}
with open(os.path.join('../data',args.ds,args.origin2),'r') as f:
    lines=f.readlines()
    for line_index in trange(len(lines),desc="combine targets"):
        if(line_index==0):continue
        line=lines[line_index]
        s_id, sentence, target_tags, opinion_words_tags = line.split('\t')
        if(sentence not in sentence_dic):
            sentence_dic[sentence]=[s_id,target_tags]
        else:
            sentence_dic[sentence][1]=combine_tar(sentence_dic[sentence][1],target_tags)
            
with open(os.path.join('./data', args.ds, args.dest2), 'w') as f:
    f.write("s_id\tsentence\ttarget_tags\n")
    for sentence in sentence_dic:
        f.write(sentence_dic[sentence][0]+'\t'+sentence+'\t'+sentence_dic[sentence][1]+'\n')

