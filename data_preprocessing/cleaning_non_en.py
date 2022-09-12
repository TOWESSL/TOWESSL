import json
import os
import string
from tqdm import tqdm
from langdetect import detect

# Clean non-en data for amazon
f2=open(os.path.join('./common_data/amazon','en_Electronics_5.json'),'w') 
with open(os.path.join('./common_data/amazon','Electronics_5.json'),'r') as f:
    for line in tqdm(f.readlines(),desc='pre amazon Electronics_5.json'):
        dic = json.loads(line)
        if('reviewText' not in dic):continue
        text=dic['reviewText']
        try:
            lan=detect(text)
        except:
            lan='err'
        if( lan!='en'):continue
        json.dump(dic,f2)
        f2.write('\n')
f2.close()

f2=open(os.path.join('./common_data/yelp','en_yelp_academic_dataset_tip.json'),'w')
with open(os.path.join('./common_data/yelp','yelp_academic_dataset_tip.json'),'r') as f:
    for line in tqdm(f.readlines(),desc='pre yelp yelp_academic_dataset_tip.json'):
        dic = json.loads(line)
        if('text' not in dic):continue
        text=dic['text']
        try:
            lan=detect(text)
        except:
            lan='err'
        if( lan!='en'):continue
        json.dump(dic,f2)
        f2.write('\n')
f2.close()