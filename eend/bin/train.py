#!/usr/bin/env python3

# Copyright 2019 Hitachi, Ltd. (author: Yusuke Fujita)
# Licensed under the MIT license.
#
import yamlargparse
import os

parser = yamlargparse.ArgumentParser(description='EEND training')
parser.add_argument('-c', '--config', help='config file path',
                    action=yamlargparse.ActionConfigFile)
parser.add_argument('train_data_dir',
                    help='kaldi-style data dir used for training.')
parser.add_argument('valid_data_dir',
                    help='kaldi-style data dir used for validation.')
parser.add_argument('model_save_dir',
                    help='output model_save_dirdirectory which model file will be saved in.')
parser.add_argument('--model-type', default='Transformer',
                    help='Type of model (Transformer)')
parser.add_argument('--initmodel', '-m', default='',
                    help='Initialize the model from given file')
parser.add_argument('--resume', '-r', default='',
                    help='Resume the optimization from snapshot')
parser.add_argument('--gpu', '-g', default=-1, type=int,
                    help='GPU ID (negative value indicates CPU)')
parser.add_argument('--max-epochs', default=20, type=int,
                    help='Max. number of epochs to train')
parser.add_argument('--input-transform', default='',
                    choices=['', 'log', 'logmel', 'logmel23', 'logmel23_mn',
                             'logmel23_mvn', 'logmel23_swn'],
                    help='input transform')
parser.add_argument('--lr', default=0.001, type=float)
parser.add_argument('--optimizer', default='adam', type=str)
parser.add_argument('--num-speakers', default=2, type=int)
parser.add_argument('--gradclip', default=-1, type=int,
                    help='gradient clipping. if < 0, no clipping')
parser.add_argument('--num-frames', default=2000, type=int,
                    help='number of frames in one utterance')
parser.add_argument('--batchsize', default=1, type=int,
                    help='number of utterances in one batch')
parser.add_argument('--label-delay', default=0, type=int,
                    help='number of frames delayed from original labels'
                         ' for uni-directional rnn to see in the future')
parser.add_argument('--hidden-size', default=256, type=int,
                    help='number of lstm output nodes')
parser.add_argument('--context-size', default=0, type=int)
parser.add_argument('--subsampling', default=1, type=int)
parser.add_argument('--frame-size', default=1024, type=int)
parser.add_argument('--frame-shift', default=256, type=int)
parser.add_argument('--sampling-rate', default=16000, type=int)
parser.add_argument('--noam-warmup-steps', default=25000, type=float)
parser.add_argument('--transformer-encoder-n-heads', default=4, type=int)
parser.add_argument('--transformer-encoder-n-layers', default=2, type=int)
parser.add_argument('--transformer-encoder-dropout', default=0.1, type=float)
parser.add_argument('--gradient-accumulation-steps', default=1, type=int)
parser.add_argument('--seed', default=777, type=int)
args = parser.parse_args()

if not os.path.exists(args.model_save_dir):
    os.makedirs(args.model_save_dir)

from eend.pytorch_backend.train import train
train(args)
