import networks
import torch
import train
from utils import *

pad_i=0
tag2id = {'B': 1, 'I': 2, 'O': 3 ,'X': 4, 'SEP':5, 'CLS': 6}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class NeuralTagger():  # Neural network method
    def __init__(self):
        self.word_embed_dim = 300
        self.hidden_size = 128
        self.vocab_size = 100
        self.output_size = 3
        pass

    def train_from_data(self, train_raw_data,test_raw_data, args):
        # W: init embedding
        self.hidden_size = args.n_hidden
        self.output_size = len(tag2id)+1 # 0 for padding

        self.tagger = networks.Pos_model(self.word_embed_dim, self.output_size, self.vocab_size, args)

        train_texts, train_targets, train_labels = train_raw_data
        train_iter, dev_iter = create_labeled_dataset(train_texts, train_targets, train_labels, args, train=True)
        test_texts, test_targets, test_labels = test_raw_data
        test_iter = create_labeled_dataset(test_texts, test_targets, test_labels, args, train=False)

        print('train')
        train.train(self.tagger, train_iter, dev_iter,test_iter, args=args)

        pass


