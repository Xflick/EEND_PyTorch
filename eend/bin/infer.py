#!/usr/bin/env python3
#
# Copyright 2019 Hitachi, Ltd. (author: Yusuke Fujita)
# Licensed under the MIT license.
#
import os
import yamlargparse

parser = yamlargparse.ArgumentParser(description='decoding')
parser.add_argument('-c', '--config', help='config file path',
                    action=yamlargparse.ActionConfigFile)
parser.add_argument('data_dir',
                    help='kaldi-style data dir')
parser.add_argument('model_file',
                    help='best.nnet')
parser.add_argument('out_dir',
                    help='output directory.')
parser.add_argument('--model_type', default='Transformer', type=str)
parser.add_argument('--gpu', type=int, default=-1)
parser.add_argument('--num-speakers', type=int, default=2)
parser.add_argument('--input-transform', default='',
                    choices=['', 'log', 'logmel',
                             'logmel23', 'logmel23_swn', 'logmel23_mn'],
                    help='input transform')
parser.add_argument('--label-delay', default=0, type=int,
                    help='number of frames delayed from original labels'
                         ' for uni-directional rnn to see in the future')
parser.add_argument('--hidden-size', default=256, type=int)
parser.add_argument('--chunk-size', default=2000, type=int,
                    help='input is chunked with this size')
parser.add_argument('--context-size', default=0, type=int,
                    help='frame splicing')
parser.add_argument('--subsampling', default=1, type=int)
parser.add_argument('--sampling-rate', default=16000, type=int,
                    help='sampling rate')
parser.add_argument('--frame-size', default=1024, type=int,
                    help='frame size')
parser.add_argument('--frame-shift', default=256, type=int,
                    help='frame shift')
parser.add_argument('--transformer-encoder-n-heads', default=4, type=int)
parser.add_argument('--transformer-encoder-n-layers', default=2, type=int)
parser.add_argument('--save-attention-weight', default=0, type=int)
args = parser.parse_args()

print(str(args))

if not os.path.exists(args.out_dir):
    os.makedirs(args.out_dir)

from eend.pytorch_backend.infer import infer
infer(args)

