import codecs
from models import *
from data_helper import load_text_target_label,load_text_target,load_text,load_aug_text
from utils import *
import os
import pickle
import random
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
# Basic Config
parser.add_argument("--batch_size", type=int, default=25)
parser.add_argument("--u_batch_size", type=int, default=100)
parser.add_argument("--eval_bs", type=int, default=200)
parser.add_argument("--training_steps", type=int, default=5000)#number of training steps
parser.add_argument("--EPOCHS", type=int, default=100)
parser.add_argument("--ds", type=str, default='14res')
parser.add_argument("--use_dev", type=str2bool, default=True)
parser.add_argument("--test_model", type=str, default=None)
parser.add_argument("--word_embed_method", type=str, default='bert')
parser.add_argument("--seed", type=int, default=100)
parser.add_argument("--cur_run_times", type=int, default=1)

# Hyperparameters
parser.add_argument("--n_hidden", type=int, default=200)
parser.add_argument("--layer_size", type=int, default=2)#layer size of BiLSTM
parser.add_argument("--optimizer", type=str,  default="AdamW")
parser.add_argument("--model", type=str, default="Pos_model")
parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--rnn_method",type=str, default='TRANSFORMER')

# Glove set
parser.add_argument("--voc_name", type=str, default='vocabulary.pkl')
parser.add_argument("--embed_name", type=str, default='embedding_table.npy')

# Train set
parser.add_argument("--ssl", type=str2bool, default=False)
parser.add_argument("--sup_gamma", type=float, default=0)
parser.add_argument("--unsup_loss", type=str, default='KLLoss')
parser.add_argument("--confidence_mask_tresh", type=float, default=0.95)
parser.add_argument("--lambda_", type=float, default=1)

# Pseudo tagging
parser.add_argument("--make_pseu_target",type=str2bool, default=False)
parser.add_argument("--random_make_pseu_target",type=str2bool, default=False)
parser.add_argument("--infer",type=str2bool, default=False)
parser.add_argument("--dump_mode",type=str, default='all')
parser.add_argument("--make_aug_target",type=str2bool, default=False)
parser.add_argument("--shield_aug",type=str, default='none')
parser.add_argument("--text_aug_mode",type=str, default='mix')
parser.add_argument("--ulb_size",type=int, default=100000)
parser.add_argument("--raw_size",type=int, default=500000)
parser.add_argument("--pseu_tresh", type=float, default=0)

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

    if not os.path.exists(os.path.join("./data/", args.ds, 'preprocessed_tar_train.tsv')):
        load_text_target(os.path.join("./data/", args.ds, 'tar_train.tsv'), os.path.join("./data/", args.ds, 'preprocessed_tar_train.tsv'))
    _,train_text, train_target = load_labeld_set(os.path.join("./data/", args.ds, 'preprocessed_tar_train.tsv'))

    if not os.path.exists(os.path.join("./data/", args.ds, 'preprocessed_tar_test.tsv')):
        load_text_target(os.path.join("./data/", args.ds, 'tar_test.tsv'), os.path.join("./data/", args.ds, 'preprocessed_tar_test.tsv'))
    
    _,test_text, test_target = load_labeld_set(os.path.join("./data/", args.ds, 'preprocessed_tar_test.tsv'))
    print(test_text[0])
    print(test_target[0])
    model = NeuralTagger()
    # model.train_from_data((train_text, train_target),(test_text, test_target), init_embedding, word2index,index2word, args)
    model.train_from_data((train_text, train_target),(test_text, test_target), None, None,None, args)

def make_pse_target():
    word2index = pickle.load(open(os.path.join('./common_data/embedding',  args.voc_name), "rb"))
    init_embedding = np.load(os.path.join('./common_data/embedding', args.embed_name))
    init_embedding = np.float32(init_embedding)
    index2word = {}
    for key, value in word2index.items():
        index2word[value] = key
    print('making pseudo target...')


    if not os.path.exists(os.path.join("./data/", args.ds, 'preprocessed_tar_aug.tsv')):
        load_aug_text_all(os.path.join("./data/", args.ds, 'unlabel_tar_train.tsv'), os.path.join("./data/", args.ds, 'preprocessed_tar_aug.tsv'), args=args)
    _,test_text, senti = load_ulb_data(os.path.join("./data/", args.ds, 'preprocessed_tar_aug.tsv'))

    # _, u_train_text, _ = load_ulb_data(os.path.join('./data',args.ds,'aug_unlabel_tar_train.tsv'))

    model = NeuralTagger()
    model.make_pseu_target(test_text, senti, init_embedding, word2index,index2word, args)
    

if __name__ == '__main__':
    if(args.make_pseu_target):
        make_pse_target()
    else:
        main()
