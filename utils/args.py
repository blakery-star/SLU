#coding=utf-8
import argparse
import sys


def init_args(params=sys.argv[1:]):
    arg_parser = argparse.ArgumentParser()
    arg_parser = add_argument_base(arg_parser)
    opt = arg_parser.parse_args(params)
    return opt


def add_argument_base(arg_parser):
    #### General configuration ####
    arg_parser.add_argument('--dataroot', default='./data', help='root of data')
    arg_parser.add_argument('--word2vec_path', default='./word2vec-768.txt', help='path of word2vector file path')
    arg_parser.add_argument('--seed', default=999, type=int, help='Random seed')
    arg_parser.add_argument('--device', type=int, default=-1, help='Use which device: -1 -> cpu ; the index of gpu o.w.')
    arg_parser.add_argument('--testing', action='store_true', help='training or evaluation mode')
    arg_parser.add_argument('--model', default="baseline", help='Model for tagging, baseline/bert')
    arg_parser.add_argument('--decode', default="baseline", help='Mode of tagging, baseline/onei/newdecode ')
    arg_parser.add_argument('--train_data', default="asr", help='Which data for training, manu/asr/MacBERT/sound/Ernie + _his or not')
    arg_parser.add_argument('--dev_data', default="asr", help='Which data for testing, asr/MacBERT/sound/Ernie')


    # For CSC (Denoising the results of ASR) #
    arg_parser.add_argument('--csc_model', default='MacBERT', 
                            help='Model for CSC. Muse be one of the following: \'MacBERT\',\'Ernie\',\'sound\'')
    arg_parser.add_argument('--csc_pretrained', type=str, default=None, help='Pretrained model for CSC.')
    arg_parser.add_argument('--csc_train', action='store_true', default=False)
    arg_parser.add_argument('--use_history', action='store_true', default=False)
    arg_parser.add_argument('--csc_save', action='store_true', default=False)

    #### Training Hyperparams ####
    arg_parser.add_argument('--batch_size', default=64, type=int, help='Batch size')
    arg_parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    arg_parser.add_argument('--max_epoch', type=int, default=100, help='terminate after maximum epochs')
    #### Common Encoder Hyperparams ####
    arg_parser.add_argument('--encoder_cell', default='LSTM', choices=['LSTM', 'GRU', 'RNN'], help='root of data')
    arg_parser.add_argument('--dropout', type=float, default=0.2, help='feature dropout rate')
    arg_parser.add_argument('--embed_size', default=768, type=int, help='Size of word embeddings')
    arg_parser.add_argument('--hidden_size', default=512, type=int, help='hidden size')
    arg_parser.add_argument('--num_layer', default=2, type=int, help='number of layer')

    return arg_parser