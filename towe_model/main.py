import codecs
from models import *
from data_helper import load_text_target_label
from utils import *
import os
import math
import random
import pickle
import numpy as np

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1','True'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0','False'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--confidence_mask_tresh", type=float, default=0.95)

parser.add_argument("--batch_size", type=int, default=10)
parser.add_argument("--u_batch_size", type=int, default=100)
parser.add_argument("--seed", type=int, default=100)
parser.add_argument("--eval_bs", type=int, default=200)
# parser.add_argument("--training_steps", type=int, default=5000)#number of training step
parser.add_argument("--epochs", type=int, default=50)

parser.add_argument("--n_hidden", type=int, default=200)
parser.add_argument("--layer_size", type=int, default=2)

parser.add_argument("--optimizer", type=str,  default="AdamW")
parser.add_argument("--lr", type=float, default=2e-4)
parser.add_argument("--ds", type=str, default='14res')

parser.add_argument("--use_dev", type=str2bool, default=True)

parser.add_argument("--test_model", type=str, default=None)
parser.add_argument("--pos_embed", type=str, default='pos_learn')#method of fixed postitional embedding
parser.add_argument("--word_embed_method", type=str, default='bert')#bert 
parser.add_argument("--voc_name", type=str, default='vocabulary.pkl')
parser.add_argument("--embed_name", type=str, default='embedding_table.npy')

parser.add_argument("--ssl", type=str2bool, default=True)
parser.add_argument("--unsup_loss", type=str, default='KLLoss')
parser.add_argument("--lambda_", type=float, default=1)
parser.add_argument("--cur_run_times", type=int, default=1)
parser.add_argument("--ulb_size", type=int, default=100000)

parser.add_argument("--infer",type=str2bool, default=False)
parser.add_argument("--dump_mode",type=str, default='all')

parser.add_argument("--unsup_loss_mode",type=str, default='none')
parser.add_argument("--small_bias", type=float, default=1e-6)
parser.add_argument("--tau", type=float, default=1)

parser.add_argument("--rnn_method", type=str, default='TRANSFORMER')
parser.add_argument("--distance_method", type=str, default='dep_dis')
parser.add_argument("--latent_method", type=str, default='dis')

parser.add_argument("--senti", type=str2bool, default=True)
parser.add_argument("--senti_model", type=str)
parser.add_argument("--senti_lr", type=float, default=1e-5)
parser.add_argument("--senti_thr", type=float, default=0.95)
parser.add_argument("--senti_iter", type=int, default=3000)
parser.add_argument("--senti_batch_size", type=int, default=128)
parser.add_argument("--strategy", type=int, default=2)



args = parser.parse_args()

# torch.set_printoptions(profile="full")
print(args)
pad_i=0
def init_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
init_seed(args.seed)


def main():
    
    # load train data
    print('loading train data...')
    if not os.path.exists(os.path.join('../data/', args.ds, 'preprocessed_train.tsv')):
        load_text_target_label(os.path.join("../data/", args.ds, 'train.tsv'), os.path.join('../data/', args.ds, 'preprocessed_train.tsv'))
    _,train_text, train_target,train_label = load_labeled_set(os.path.join('../data/', args.ds, 'preprocessed_train.tsv'))

    if not os.path.exists(os.path.join('../data/', args.ds, 'preprocessed_test.tsv')):
        load_text_target_label(os.path.join("../data/", args.ds, 'test.tsv'), os.path.join('../data/', args.ds, 'preprocessed_test.tsv'))
    _,test_text, test_target,test_label = load_labeled_set(os.path.join('../data/', args.ds, 'preprocessed_test.tsv'))

    model = NeuralTagger()
    model.train_from_data((train_text, train_target,train_label),(test_text, test_target,test_label), args)

if __name__ == '__main__':
    main()
