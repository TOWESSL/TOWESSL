# Repo for Exploiting Unlabeled Data for Target-Oriented Opinion Words Extraction
Accepted by COLING2022

*** 
## Quick Start
### Google Drive
https://drive.google.com/drive/folders/1gF22rNVSaP9TERcxyxt1ouq7UT5cdKEo?usp=sharing

### Preprocessed Data & Pretrained Model
Why we provide these files?   
=> The preprocessing of meta data and pretraining of 1st stage Pseudo Labeling is tedious. To avoid boring work, we provide the files we used in our experiments.  
Here are the reasons:
- The meta data of Unlabeled data is extremely large. 
- The preprocessing of data is complex.
- Towe-ssl is the main point of this paper, so we offer a quick start way to test our method.
- The pretrained models and data files can provide the reproducibility of our paper.

### Folder Structure
1. Please download the pretrained files and unzip as the following structure:  
```
|  
|-data  
|   |  
|   |-14res
|       |
|       |-...(many preprocessed data files)
|   |-15res
|   |-16res
|   |-14lap
|
|-towe_model
    |
    |-backup
    |   |
    |   |-senti_14lap_bert.pt
    |   |-senti_14res_bert.pt
    |   |-senti_15res_bert.pt
    |   |-senti_16res_bert.pt
    |
    |-main.py
    |-.....(other .py)
```

### Scripts
To make the training easy, we provide different scripts to make the experiments convenience.  
Please download and unzip the scripts at the folder you like.  
Here we take senti_14res.sh for instance, which means the experiments for 14res dataset. 
```
# Enter the towe-ssl folder
bash [the path to senti_14res.sh] 1 # 1 is the order of the GPU you want to use.
```

This is the source code of the senti_14res.sh.
```
gpu=$1
ds=14res

export CUDA_VISIBLE_DEVICES=${gpu}
mkdir logs_100_senti 

for seed in 1 2 3
do
    for senti_thr in 0.9 0.7 0.5
    do
        for conf_thr in 0.9 0.7 0.5
        do
            python main.py --batch_size 16 --u_batch_size 96 --ds $ds --cur_run_times 1 --confidence_mask_tresh $conf_thr --senti_model senti_${ds}_bert.pt --senti_thr $senti_thr --epochs 100 --strategy 2 --seed $seed > ./logs_100_senti/senti_${ds}_${conf_thr}_${senti_thr}_${seed}.log 
        done
    done 
done
```

***
## Experiments Logs
We also provide the meta logs of our experiments to show the reliability of this paper.  
You can unzip the `exp_results.rar` to check that.

## Pseudo Labeling Model
The pretrained pseudo labeling model is provided as well, though we dont recommend you to reproduce the experiment from scretch.  

***
## Train Details
We train our model on single 3090GPU.  
The memeory cost is 22K mb and time cost is about 80mins / 100epochs.

***
## Datasets

### TOWE 
The datasets of TOWE.  
14lap~16res: https://github.com/NJUNLP/TOWE

### Amazon
For 14lap dataset unlabeled data.  
http://deepyeti.ucsd.edu/jianmo/amazon/categoryFilesSmall/Electronics_5.json.gz  

(if you can't open this url,please goto https://nijianmo.github.io/amazon/index.html and click Electronics 5-core (6,739,590 reviews) in "Small" subsets for experimentation)

### YELP
For 14~16res datasets unlabeled data.   
https://www.kaggle.com/yelp-dataset/yelp-dataset

***

## Full Script
This part shows the full work flow of this work. From original data to the final result.  
If you want to run the code fast, we recommand you use the preprocessed data and pretrained model we provided.

### Data Preparation
1. Clening non English data from Amazon and Yelp:  
``` 
python ./data_preprocessing/cleaning_non_en.py
```
2. Preprocessing raw data.  
```
python ./pre_amazon_yelp_raw.py --balance False
```

### Pseudo Opinion Word Labeling
3. Prepare data for POWL. --ds: 14lap or 14~16res. (The same below)
```
python ./data_preprocessing/pre_tar_label.py  --ds 14lap
```  
4.  Select unlabeled data.  
``` 
python ./data_preprocessing/choose_uda_tar_size.py --aug_size 3000000 --ds 14lap --raw amazon
```
5. Do the data augmentation.
``` 
python ./data_preprocessing/pre_pseudo_tar_aug.py --aug_mode mix --ds 14lap
``` 
6. The final preprocessing for BERT.
```
python ./pseudo_labeling/pre_bert_aug.py --ds 14lap --cur_run_times 1
```
7. Pretrain Pseudo Opinion Labeling model with Supervised Learning.
``` 
python main.py --seed 1 --ds 14lap --ssl False 
```
8. Labeling pseudo opinion words with pretrained models.
```
python main.py --make_pseu_target True --eval_bs 512 --test_model tar_14lap_bert_1.pt --cur_run_times 1 --ds 14lap
```

### TOWE model
9. Split unlabeled data for towe model.
```
python ./data_preprocessing/split_unlabel.py --ds 14lap --cur_run_times 1
```
10. Do the data augmentation for towe model.
```
python ./data_preprocessing/pre_towe_aug.py --aug_mode mix --ds 14lap --cur_run_times 1
python ./towe_model/pre_bert_aug.py --ds 14lap --cur_time 1
```
11. Train towe model
```
python main.py --batch_size 16 --u_batch_size 96 --ds 14lap --cur_run_times 1 --confidence_mask_tresh 0.9 --senti_model senti_14lap_bert.pt --senti_thr 0.9 --ulb_size 200000 --epochs 100 --strategy 2 --ssl True --seed 0
```

*** 
## 