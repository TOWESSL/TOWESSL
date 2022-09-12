from csv import excel_tab
import torch
from torch import nn
from data_helper import *
import time
from tqdm import tqdm,trange
import numpy as np
import torch.nn.functional as F
import copy
import matplotlib
import pickle
import os
import statistics
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from evaluate import score_BIO
import math
from numpy import linalg as la
import random
import torchtext.data as data
from utils import * 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pad_i=0
tag2id = {'B': 1, 'I': 2, 'O': 3 ,'X': 4, 'SEP':5, 'CLS': 6}
id2tag = {1: 'B', 2:'I', 3:'O', 4:'X', 5:'SEP', 6:'CLS'}

def TSA(output_p,labels,n_epochs,epoch,step,total_step,args):
    loss=F.nll_loss(output_p,labels,ignore_index=pad_i,reduction='none')
    prob=torch.exp(-loss)
    current_step=((epoch)*total_step+step)/(n_epochs*total_step)
    if(args.tsa_mode=='linear'):
        alpha=current_step
    elif(args.tsa_mode=='exp'):
        alpha=math.exp(5*(current_step-1))
    elif(args.tsa_mode=='log'):
        alpha=1-math.exp(-current_step*5)
    class_num=len(tag2id)
    n_t=alpha*(1-1/class_num)+1/class_num
    larger_than_threshold = prob > n_t
    loss_mask = torch.ones_like(labels, dtype=torch.float32) * (1 - larger_than_threshold.type(torch.float32)).detach()
    loss = torch.sum(loss * loss_mask, dim=-1)
    loss=loss/output_p.shape[0]
    return loss

def unsuploss(text,aug_text,prob1,logprob1,prob2,logprob2,args):#prob1:input prob2:target
    if(args.unsup_loss=='KLLoss'):
        loss=nn.KLDivLoss(reduction='none')(logprob1,prob2)
    elif(args.unsup_loss=='MSELoss'):
        loss=nn.MSELoss(reduction='none')(prob1,prob2)
    return loss.sum()/prob1.shape[0]

def confidence_mask_unsuploss(text,aug_text,prob1,logprob1,prob2,logprob2,args):
    max_prob,max_idx=torch.max(prob2,dim=-1)
    loss=F.nll_loss(logprob1,max_idx,reduction='none')
    loss_mask= (max_prob>=args.confidence_mask_tresh).float()
    loss_mask= loss_mask.unsqueeze(dim=1)
    loss=loss.reshape(-1)
    loss_mask=loss_mask.reshape(-1).detach()
    loss = torch.sum(loss  * loss_mask, dim=-1)
    mask_ratio=loss_mask.sum().item()/prob2.shape[0]
    if(loss_mask.sum()>0):
        return loss/loss_mask.sum(),1-mask_ratio
    return loss,1
 
def train(model, train_iter, dev_iter,test_iter,W,word2index,index2word,args):
    time_stamp = time.asctime().replace(':', '_').split()
    print(time_stamp) 

    n_epochs = int(args.training_steps/len(train_iter))
    learning_rate = args.lr
    model.to(device)

    bert_params = list(map(id, model.bert_model.parameters()))
    base_params = filter(lambda p: id(p) not in bert_params,model.parameters())
    params = [{'params': base_params},
              {'params': model.bert_model.parameters(), 'lr': 1e-5}]
    if args.optimizer == "AdamW":
        if(args.word_embed_method=='glove'):
            learning_rate=1e-3        
        optimizer = torch.optim.AdamW(params,lr=learning_rate) 

    start = time.time()
    best_score = 0
    train_bert=True

    # Prepare unlabeled data 
    if args.ssl == True:
        u_train_set = make_u_batch_iter(args)
        u_train_iter = data.Iterator(u_train_set, 
                                batch_size = args.u_batch_size, 
                                repeat=False,
                                shuffle=True,
                                device=device
                                )
    

    for epoch in range(int(n_epochs)):
        cur_lr=optimizer.state_dict()['param_groups'][0]['lr']
        print('learning rate',cur_lr)

        loss_sum = 0
        loss1_sum=0
        loss2_sum=0
        model.zero_grad()

        for step, batch in enumerate(train_iter):
            model.train()
            loss=0
            loss1=0
            loss2=0
            mask_ratio_mean=[]
            _,output_logp = model(batch,train_bert)
            output_target=output_logp.max(dim=-1)[1]
            
            if(args.ssl):
                if(args.sup_gamma==0):
                    loss1=F.nll_loss(output_logp.view(-1, model.output_size),batch.target.view(-1),ignore_index=pad_i)
                else:
                    loss1=F.nll_loss(output_logp.view(-1, model.output_size),batch.target.view(-1),ignore_index=pad_i,reduction='none')
                    prob=torch.exp(-loss1)
                    prob_gamma=(1-prob).pow(args.sup_gamma).detach()
                    loss1=loss1*prob_gamma
                    loss1=loss1.sum()/output_logp.shape[0]

                try:            
                    u_batch=next(iter(u_train_iter))
                except:
                    u_train_iter = data.Iterator(u_train_set, 
                                batch_size = args.u_batch_size, 
                                repeat=False,
                                shuffle=True,
                                device=device
                                )
                    u_batch = next(iter(u_train_iter))

                # logits_u
                u_output_p,u_output_logp = model(u_batch,train_bert)
                # logits_aug
                u_aug_output_p,u_aug_output_logp = model(u_batch,train_bert,aug=True)

                u_output_p=u_output_p.detach()
                u_output_logp=u_output_logp.detach()
                
                loss2,mask_ratio=confidence_mask_unsuploss(None,None,u_aug_output_p.view(-1, model.output_size),u_aug_output_logp.view(-1, model.output_size),u_output_p.view(-1, model.output_size),u_output_logp.view(-1, model.output_size),args)
                mask_ratio_mean.append(mask_ratio)
                loss=loss1+args.lambda_*loss2
            
            else:
                if(args.sup_gamma==0):
                    loss1=F.nll_loss(output_logp.view(-1, model.output_size),batch.target.view(-1),ignore_index=pad_i)
                else:
                    loss1=F.nll_loss(output_logp.view(-1, model.output_size),batch.target.view(-1),ignore_index=pad_i,reduction='none')
                    prob=torch.exp(-loss1)
                    prob_gamma=(1-prob).pow(args.sup_gamma).detach()
                    loss1=loss1*prob_gamma
                    loss1=loss1.sum()/output_logp.shape[0]
                loss=loss1
                mask_ratio_mean.append(0)
            
            loss.backward()
            optimizer.step()
            model.zero_grad()

            loss_sum += loss.item()
            loss1_sum += loss1.item()
            if(args.ssl):
                loss2_sum += loss2.item()
        print("Epoch:%d" % epoch)
        with torch.no_grad():
            eval_dict = eval(model, dev_iter, W,word2index,index2word,args)
        print("DEV: p:%.4f, r:%.4f, f:%.4f, loss:%.4f" % (eval_dict['precision'], eval_dict['recall'], eval_dict['f1'],eval_dict['loss']))
        with torch.no_grad():
            test_dict = eval(model, test_iter, W,word2index,index2word,args)
        print("TEST: p:%.4f, r:%.4f, f:%.4f, loss:%.4f" % (test_dict['precision'], test_dict['recall'], test_dict['f1'],test_dict['loss']))
        if eval_dict['main_metric'] >= best_score:
            best_score = eval_dict['main_metric']
            max_print = ("Epoch%d\n" % epoch
                         + "DEV: p:%.4f, r:%.4f, f:%.4f, loss:%.4f\n" % (eval_dict['precision'], eval_dict['recall'], eval_dict['f1'],eval_dict['loss'])
                         + "TEST: p:%.4f, r:%.4f, f:%.4f, loss:%.4f\n" % (test_dict['precision'], test_dict['recall'], test_dict['f1'],test_dict['loss']))
            best_dict = copy.deepcopy(model.state_dict())

        print("Epoch: %d, total loss: %.4f, sup loss: %.4f, unsup loss: %.4f, mask_ratio %.4f" % (epoch, loss_sum,loss1_sum,loss2_sum, statistics.mean(mask_ratio_mean)))

    if not os.path.exists('backup/'):
        os.mkdir('backup')
    best_model = "backup/tar_%s_%s_%d" % (args.ds,args.word_embed_method,args.cur_run_times)
    best_model=best_model.replace('.','')
    best_model=best_model+'.pt'
    torch.save(best_dict, best_model)
    print("Best Result:")
    print(max_print)
    time_stamp = time.asctime().replace(':', '_').split()
    print(time_stamp)
    return best_dict


def category_from_output(output):
    top_n, top_i = output.topk(1) # Tensor out of Variable with .data
    # print(top_i)
    category_i = top_i.view(output.size()[0], -1).detach().tolist()
    return category_i

def random_category_from_output(text):
    num=text.shape[0]
    length=text.shape[1]
    res=np.random.choice([1,2,3],(num,length),p=[0.85,0.1,0.05]).tolist()
    return res




def translate_text(text,index2word):
    res=[]
    for i in range(text.shape[0]):
        tmp=[]
        for j in range(text.shape[1]):
            tmp.append(index2word[text[i,j].item()])
        res.append(tmp)
    return res
def translate_senti(senti):
    res=[]
    for i in range(senti.shape[0]):
        res.append(id2senti[senti[i].item()])
    return res
def fix_bio(l):
    res=copy.deepcopy(l)
    for i in range(len(l)):
        for j in range(len(l[i])):
            if(l[i][j]==tag2id['B'] and l[i][j-1]==tag2id['B']):
                res[i][j] = tag2id['I']
            elif(l[i][j]==tag2id['I'] and l[i][j-1]==tag2id['O']):
               res[i][j] = tag2id['B']
            elif(l[i][j]==pad_i):
                res[i][j] = tag2id['O']

    return res
def random_make_pseu_target(model, dev_iter,W,word2index,index2word, args):
    predicted_result = []
    text_list=[]
    senti_list=[]
    d_count=0
    for eval_batch in dev_iter:
        d_count+=1
        category_i_p = random_category_from_output(eval_batch.text[0])
        text_list.extend(translate_text(eval_batch.text[0],index2word))
        predicted_result.extend(category_i_p)
        senti_list.extend(translate_senti(eval_batch.senti))
    print(predicted_result[0])
    print(len(text_list))

    predicted_result=fix_bio(predicted_result)
    for i in trange(len(text_list)):
        for j in range(len(text_list[i])):
            if(text_list[i][j]==index2word[pad_i]):
                text_list[i]=text_list[i][:j]
                predicted_result[i]=predicted_result[i][:j]
                break
            predicted_result[i][j]=text_list[i][j]+'\\'+id2tag[predicted_result[i][j]]
    print(text_list[0])
    print(predicted_result[0])
    return text_list,predicted_result,senti_list

def make_pseu_target(model, dev_iter,W,word2index,index2word, args):

    logit_list = []
    predicted_result = []
    text_list=[]
    d_count=0
    senti_list = []

    for eval_batch in dev_iter:
        model.eval()
        d_count+=1
        _,output= model(eval_batch,train_bert=True)
        logits = output.detach()
        logit_list.extend(logits.tolist())
        category_i_p = category_from_output(output)
        text_list.extend(translate_text(eval_batch.text[0],index2word))
        predicted_result.extend(category_i_p)
        senti_list.extend(eval_batch.senti.tolist())
    
    assert len(senti_list) == len(text_list)

    remove_list=[]
    for i in range(len(logit_list)):
        for j in range(len(logit_list[i])):
            if(text_list[i][j]==index2word[pad_i]):
                break
            if(args.pseu_tresh!=0 and max(logit_list[i][j])<math.log(args.pseu_tresh)):
                remove_list.append(i)
                break

    remove_list.reverse()
    
    for i in remove_list:
        del text_list[i]
        del predicted_result[i]
        del senti_list[i]

    print('remove size',len(remove_list))
    print(predicted_result[0])
    print(len(text_list))

    predicted_result=fix_bio(predicted_result)
    for i in trange(len(text_list)):
        for j in range(len(text_list[i])):
            if(text_list[i][j]==index2word[pad_i]):
                text_list[i]=text_list[i][:j] 
                predicted_result[i]=predicted_result[i][:j]
                break
            predicted_result[i][j]=text_list[i][j]+'\\'+id2tag[predicted_result[i][j]]
    print(text_list[0])
    print(predicted_result[0])
    return text_list,predicted_result,senti_list

def eval(model, dev_iter,W,word2index,index2word, args, dump_mode=None, dump_file=""):
    if(args.infer):
        dump_mode=args.dump_mode
        dump_file='inferlogs/'+args.test_model.replace('.pt','.txt')
    logit_list = []
    target_list=[]
    labels_eval_list = []
    predicted_result = []
    text_list=[]
    loss=0
    for eval_batch in dev_iter:
        model.eval()
        _,output= model(eval_batch,train_bert=True)
        #print(output.shape)
        loss+=F.nll_loss(output.view(-1, model.output_size),eval_batch.target.view(-1),ignore_index=pad_i).item()
        category_i_p = category_from_output(output)

        predicted_result.extend(category_i_p)
        target_ids = eval_batch.target
        target_list.extend(target_ids.tolist())
        logits = output.detach()
        logit_list.extend(logits.tolist())
        #print(eval_batch.text[0].tolist())
        #print(category_i_p)
        #print(label_ids.tolist())

    eval_dict = {}
    score_dict = score_BIO(predicted_result, target_list, ignore_index=pad_i)

    eval_dict.update(score_dict)
    eval_dict['main_metric'] = score_dict['f1']
    eval_dict['loss']=loss
    
    return eval_dict
