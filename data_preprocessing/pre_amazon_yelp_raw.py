import json
import os
import string
import random
from tqdm import tqdm
from langdetect import detect
import numpy as np

random.seed(1)
np.random.seed(1)
res=[]
len_tresh=150
senti_tresh=4

p_dic=['\'','-']
def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1','True'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0','False'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--balance", type=str2bool, default=True)
parser.add_argument("--balance_size", type=int, default=250000)
parser.add_argument("--dataset", type=str, default="yelp")

"""
Create raw data based on En_xx
data: text by \s
number: neg&pos: 100k * 2
"""
args = parser.parse_args()

if args.dataset == 'yelp':
    res=[]
    with open(os.path.join('./common_data/yelp','en_yelp_academic_dataset_review.json'),'r') as f:
        pos_res=[]
        neg_res=[]
        pos_num=0
        neg_num=0
        count = 0
        for line in tqdm(f.readlines(),desc='pre yelp yelp_academic_dataset_review.json'):
            count += 1
            if count == 7000000:
                break
            dic = json.loads(line)
            if('text' not in dic):continue
            text=dic['text']
            senti=dic['stars']
            for p in string.punctuation:
                if(p in p_dic):continue
                text=text.replace(p,' '+p)
                text=text.replace(p,p+' ')
            text=text.replace('\t','').replace('\n','').replace('\r','').replace('\f','')
            text=text.split()
            while('' in text):
                text=text.remove('')
            text=' '.join(text)

            if(senti>=senti_tresh):
                text=text+'\t'+'positive'+'\n'
            else:
                text=text+'\t'+'negtive'+'\n'
            
            if(len(text)>1 and len(text)<=len_tresh):
                if(senti>=senti_tresh):
                    pos_res.append(text)
                    pos_num+=1
                else:
                    neg_res.append(text)
                    neg_num+=1
                res.append(text)
        
        random.shuffle(pos_res)
        random.shuffle(neg_res)
        if(args.balance==True):
            pos_res=pos_res[:args.balance_size]
            neg_res=neg_res[:args.balance_size]
        res=pos_res+neg_res
        random.shuffle(res)

    print('total:yelp review,pos,neg,all',pos_num,neg_num,pos_num+neg_num)
    print('chose:yelp review,pos,neg,all',len(pos_res),len(neg_res),len(res))

    with open(os.path.join('./common_data/yelp','senti_yelp_academic_dataset_review.txt'),'w') as f:
        f.writelines(res)

elif args.dataset == 'lap':
    res = []
    with open(os.path.join('./common_data/lap','amazon_lap_ulb.txt'),'r') as f:
        pos_num=0
        neg_num=0
        pos_res=[]
        neg_res=[]

        for line in tqdm(f.readlines(),desc='pre amazon_lap_ulb.txt'):
            text, senti = line.split('\t')
            if senti == '1':
                text = text + '\t' + 'positive' + '\n'
                pos_res.append(text)
                pos_num += 1
            else:
                text = text + '\t' + 'negative' + '\n'
                neg_res.append(text)
                neg_num += 1
            
            if args.balance == True:
                pos_res=pos_res[:args.balance_size]
                neg_res=neg_res[:args.balance_size]
        res=pos_res+neg_res
        random.shuffle(res)
            
        print('total:amazon review,pos,neg,all',pos_num,neg_num,pos_num+neg_num)
        print('chose:amazon review,pos,neg,all',len(pos_res),len(neg_res),len(res))
    with open(os.path.join('./common_data/lap','senti_reviews_Electronics_5.txt'),'w') as f:
        f.writelines(res)


