from nltk.corpus.reader.chasen import test
from nltk.util import pr
import torch
from torch import nn
import time

from torch._C import dtype
from utils import *
from tqdm import tqdm,trange
import numpy as np
import torch.nn.functional as F
import copy
import matplotlib
import pickle
import os
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from evaluate import score_BIO
import math
from numpy import linalg as la
import random
import operator
import torchtext.data as data
import statistics
from transformers import AdamW
import json
from senti_network import Senti_model

pad_i = 0
tag2id = {'B': 1, 'I': 2, 'O': 3 ,'X': 4, 'SEP':5, 'CLS': 6}
id2tag = {0:'P',1: 'B', 2:'I', 3:'O',4:'X', 5:'SEP', 6:'CLS'}
taroth2id = {'B': 1+1, 'I': 1+1, 'O': 0+1}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def confidence_mask_unsuploss(u_batch, prob1,logprob1,prob2,logprob2,logits,args, gloabal_conf, pad_mask):
    prob2=prob2.detach()
    logprob2=logprob2.detach()
    max_prob,max_idx=torch.max(prob2,dim=-1)

    strategy = args.strategy

    if strategy != 0 and strategy != 4:
        gloabal_thr = args.senti_thr
        gloabal_conf = gloabal_conf.reshape(-1)
        gloabal_mask = (gloabal_conf >= gloabal_thr).float()
        max_prob = gloabal_mask * max_prob
    elif strategy == 0:
        pass
    elif strategy == 4:
        # UDA
        pass

    loss_mask= (max_prob>=args.confidence_mask_tresh).float()
    loss_mask= loss_mask.unsqueeze(dim=1)

    if strategy != 4:        
        loss=F.nll_loss(logprob1,max_idx,reduction='none')
        
    else:
        prob2 = prob2 / 0.4 #T = 0.4
        prob2 = prob2.detach()
        loss = F.kl_div(logprob1, prob2, reduce="none")
    loss=loss.reshape(-1)
    loss_mask=loss_mask.reshape(-1).detach()
    pad_mask = pad_mask.reshape(-1).detach()
    loss = torch.sum(loss  * loss_mask * pad_mask, dim=-1)

    # mask ratio:
    total_num = pad_mask.sum().item()
    if strategy != 0 and strategy != 4:
        remained_num = (loss_mask.bool() & pad_mask.bool()).sum().item()
        mask_ratio = 1 - (remained_num/total_num)
    else:
        remained_num = loss_mask.sum().item()
        mask_ratio = 1 - (remained_num/total_num)
        # mask_ratio=loss_mask.sum().item()/prob2.shape[0]

    if(loss_mask.sum()>0):
        return loss/loss_mask.sum(),1-mask_ratio
    
    return loss,1



def train(model, train_iter, dev_iter,test_iter,args):
    time_stamp = time.asctime().replace(':', '_').split()
    print(time_stamp)

    learning_rate = args.lr
    model.to(device)
    bert_params = list(map(id, model.bert_model.parameters()))
    base_params = filter(lambda p: id(p) not in bert_params,model.parameters())
    
    params = [{'params': base_params},
              {'params': model.bert_model.parameters(), 'lr': 2e-5},
              ]

    if args.optimizer == "AdamW":
        optimizer = AdamW(params,lr=learning_rate)

    if args.optimizer == "SGD":
        if(args.word_embed_method=='glove'):
            learning_rate=1e-3 
        optimizer = torch.optim.SGD(params,lr=learning_rate) 
    
    best_score = 0
    if args.word_embed_method == 'glove':
        train_bert=False 
    elif args.word_embed_method == 'bert':
        train_bert=True

    if args.ssl:
        u_train_set = make_u_batch_iter(args)
        u_train_iter = data.Iterator(u_train_set, batch_size=args.u_batch_size, repeat=False, shuffle=True,
                                device=device)

    senti_model = Senti_model().cuda()
    # Senti Pretrained Model
    if not os.path.exists(os.path.join('./backup', args.senti_model)):
        print('Pretrain Sentiment Model')
        pretrain_senti_model(senti_model, u_train_set, args)

    print('Load pretrain')
    senti_model.load_state_dict(torch.load(os.path.join('./backup', args.senti_model)))
    senti_model.eval()

    # evaluate_senti(senti_model, u_train_set, args)
    
    # ERROR ANALYSIS
    best_err = None

    for epoch in trange(int(args.epochs)):
        loss_sum = 0
        loss1_sum=0
        loss2_sum=0
        mask_ratio_mean=[]
        model.zero_grad()

        for step, batch in enumerate(train_iter):
            model.train()
            loss=0
            loss1=0
            loss2=0
            _,_,output_logp = model(batch,train_bert)

            loss1=F.nll_loss(output_logp.view(-1, model.output_size),batch.label.view(-1),ignore_index=pad_i)

            if(args.ssl):
                try:
                    u_batch=next(iter(u_train_iter))
                except:
                    u_train_iter = data.Iterator(u_train_set, batch_size=args.u_batch_size, repeat=False, shuffle=True,device=device)
                    u_batch = next(iter(u_train_iter))

                u_logits, u_output_p,u_output_logp = model(u_batch,train_bert)
                u_aug_logits,u_aug_output_p,u_aug_output_logp = model(u_batch,train_bert,aug=True,)
                u_output_p=u_output_p.detach()
                u_output_logp=u_output_logp.detach()

                # ulb loss mask
                pad_mask = u_batch.bert_mask
                b, s  = pad_mask.shape
                if args.strategy == 0:
                    gloabal_conf=None
                elif args.strategy == 2:
                    # senti
                    _ , senti_attn = senti_model(u_batch)
                    b, s = senti_attn.shape
                    max_p, _ = torch.max(u_output_p, dim=-1) # [b, s]
                    senti_attn = senti_attn * pad_mask
                    senti_attn = F.normalize(senti_attn, p=1, dim=1)
                    gloabal_conf = (max_p * senti_attn).sum(dim=-1) # b
                    gloabal_conf = gloabal_conf.unsqueeze(-1).expand(b, s)
                elif args.strategy == 3:
                    # dis-senti
                    _ , senti_attn = senti_model(u_batch)
                    b, s = senti_attn.shape
                    dis_abs = u_batch.dis
                    c_i = 1 - (dis_abs / s)
                    alpha_i = c_i * senti_attn  * pad_mask
                    alpha_i = F.normalize(alpha_i, p=1, dim=1)
                    max_p, _ = torch.max(u_output_p, dim=-1) # [b, s]
                    gloabal_conf = (max_p * alpha_i).sum(dim=-1) # b
                    gloabal_conf = gloabal_conf.unsqueeze(-1).expand(b, s)
                elif args.strategy == 1:
                    # avg
                    max_p, _ = torch.max(u_output_p, dim=-1) # [b, s]
                    max_p = max_p * pad_mask
                    b_l = torch.sum(pad_mask, dim=1) # [b]
                    gloabal_conf = torch.sum(max_p, dim=1) / b_l
                    gloabal_conf = gloabal_conf.unsqueeze(-1).expand(b, s)
                elif args.strategy == 4:
                    # UDA
                    gloabal_conf = None 



                loss2,mask_ratio=confidence_mask_unsuploss(u_batch,u_aug_output_p.view(-1, model.output_size),u_aug_output_logp.view(-1, model.output_size),u_output_p.view(-1, model.output_size),u_output_logp.view(-1, model.output_size),u_logits,args, gloabal_conf, pad_mask)
                
                mask_ratio_mean.append(mask_ratio)

                loss = loss1 + loss2
            else:
                loss = loss1 

            loss.backward()
            optimizer.step()
            model.zero_grad()
            
            loss_sum += loss.item()
            loss1_sum += loss1.item()
            if(args.ssl):
                loss2_sum += loss2.item()

        print("Epoch:%d" % epoch)
        with torch.no_grad():
            eval_dict, _ = eval(model, dev_iter,args)
        print("DEV: p:%.4f, r:%.4f, f:%.4f, loss:%.4f" % (eval_dict['precision'], eval_dict['recall'], eval_dict['f1'], eval_dict['loss']))
        with torch.no_grad():
            test_dict, test_err = eval(model, test_iter,args)
        print("TEST: p:%.4f, r:%.4f, f:%.4f, loss:%.4f" % (test_dict['precision'], test_dict['recall'], test_dict['f1'], test_dict['loss']))
        if eval_dict['main_metric'] >= best_score:
            best_score = eval_dict['main_metric']
            max_print = ("Epoch%d\n" % epoch
                         + "DEV: p:%.4f, r:%.4f, f:%.4f, loss:%.4f\n" % (eval_dict['precision'], eval_dict['recall'], eval_dict['f1'],eval_dict['loss'])
                         + "TEST: p:%.4f, r:%.4f, f:%.4f, loss:%.4f\n" % (test_dict['precision'], test_dict['recall'], test_dict['f1'],test_dict['loss']))
            best_dict = copy.deepcopy(model.state_dict())
            best_err = test_err

        if args.ssl:
            print("Epoch: %d, total loss: %.4f, sup loss: %.4f, unsup loss: %.4f, mask_ratio %.4f " % (epoch, loss_sum,loss1_sum,loss2_sum, statistics.mean(mask_ratio_mean)))
        else:
            print("Epoch: %d, total loss: %.4f, sup loss: %.4f, unsup loss: None, mask_ratio: None" % (epoch, loss_sum,loss1_sum))

        print("best: ")
        print(max_print)

    best_model = "backup/%s_%s_%d" % (args.ds,args.word_embed_method,args.cur_run_times)
    best_model=best_model.replace('.','')
    best_model=best_model+'.pt'
    if not os.path.exists('backup/'):
        os.mkdir('backup')

    # Translate errs back to sentences and save
    with open('./'+args.ds+'_'+str(args.ssl)+'.txt', 'w', encoding='utf-8') as w:
        ids_, pred_, label_ = best_err
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        for i in range(len(ids_)):
            assert(len(ids_[i]) == len(pred_[i]))
            assert(len(ids_[i]) == len(label_[i]))
            tokens_ =  tokenizer.convert_ids_to_tokens(ids_[i])
            p_ = []
            l_ = []
            

            for j in range(len(tokens_)):
                if tokens_[j] == '[PAD]':
                    break
                p_.append(tokens_[j] + '/' + id2tag[pred_[i][j]])
                l_.append(tokens_[j] + '/' + id2tag[label_[i][j]])

            pp = ' '.join(p_)
            ll = ' '.join(l_)

            w.write(pp)
            w.write('\n')
            w.write(ll)
            w.write('\n')
            w.write('\n')


    # torch.save(best_dict, best_model)
    print("Best Result:")
    print(max_print)
    time_stamp = time.asctime().replace(':', '_').split()
    print(time_stamp)
    return best_dict


def category_from_output(output):
    top_n, top_i = output.topk(1) # Tensor out of Variable with .data
    category_i = top_i.view(output.size()[0], -1).detach().tolist()
    return category_i

def translate_text(text,index2word):
    res=[]
    for i in range(text.shape[0]):
        tmp=[]
        for j in range(text.shape[1]):
            tmp.append(index2word[text[i,j].item()])
        res.append(tmp)
    return res
def count_error(labels,predicts):
    labels=np.array(labels)
    predicts=np.array(predicts)
    l_index=np.where(labels>1)[0].tolist()
    p_index=np.where(predicts>1)[0].tolist()
    under=0
    over=0
    null=0
    other=0
    if(operator.eq(p_index,l_index)):
        return under,over,null,other
    if(len(p_index)==0):
        null=1
        return under,over,null,other
    mix_index=list(set(l_index).intersection(set(p_index)))
    if(operator.eq(p_index,mix_index)):
        under=1
        return under,over,null,other
    if(operator.eq(l_index,mix_index)):
        over=1
        return under,over,null,other
    other=1
    return under,over,null,other

def eval(model, dev_iter, args, dump_mode=None, dump_file=""):
    if(args.infer):
        dump_mode=args.dump_mode
        dump_file='inferlogs/'+args.test_model.replace('.pt','.txt')
    logit_list = []
    labels_eval_list = []
    predicted_result = []
    target_list=[]
    loss=0
    sentence_id = []
    for eval_batch in dev_iter:
        model.eval()
        _,_,output = model(eval_batch,train_bert=True)

        loss+=F.nll_loss(output.view(-1, model.output_size),eval_batch.label.view(-1),ignore_index=pad_i).item()

        target_ids = eval_batch.target
        target_list.extend(target_ids.tolist())
        
        logits = output.detach()
        logit_list.extend(logits.tolist())

        category_i_p = category_from_output(output) # B*S, 1
        predicted_result.extend(category_i_p) # total, 1
        label_ids = eval_batch.label
        labels_eval_list.extend(label_ids.tolist())
        batch_ids = eval_batch.bert_ids
        sentence_id.extend(batch_ids.tolist())

        assert len(category_i_p) == len(label_ids)

    # print(predicted_result)
    eval_dict = {}
    score_dict, errors = score_BIO(predicted_result, labels_eval_list, ignore_index=pad_i, sentence_id=sentence_id)
    eval_dict.update(score_dict)
    eval_dict['main_metric'] = score_dict['f1']
    eval_dict['loss']=loss

    return eval_dict, errors
