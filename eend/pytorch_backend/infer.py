#!/usr/bin/env python3
#
# Copyright 2019 Hitachi, Ltd. (author: Yusuke Fujita)
# Modified by: Yexin Yang
# Licensed under the MIT license.
#
import os
import h5py
import numpy as np
from scipy.ndimage import shift

import torch
import torch.nn as nn

from eend.pytorch_backend.models import TransformerModel
from eend import feature
from eend import kaldi_data


def _gen_chunk_indices(data_len, chunk_size):
    step = chunk_size
    start = 0
    while start < data_len:
        end = min(data_len, start + chunk_size)
        yield start, end
        start += step


def infer(args):
    # Prepare model
    in_size = feature.get_input_dim(
            args.frame_size,
            args.context_size,
            args.input_transform)

    if args.model_type == 'Transformer':
        model = TransformerModel(
                n_speakers=args.num_speakers,
                in_size=in_size,
                n_units=args.hidden_size,
                n_heads=args.transformer_encoder_n_heads,
                n_layers=args.transformer_encoder_n_layers,
                has_pos=False
                )
    else:
        raise ValueError('Unknown model type.')

    device = torch.device("cuda" if (torch.cuda.is_available() and args.gpu > 0) else "cpu")
    if device.type == "cuda":
        model = nn.DataParallel(model, list(range(args.gpu)))
    model = model.to(device)

    model.load_state_dict(torch.load(args.model_file))
    model.eval()

    kaldi_obj = kaldi_data.KaldiData(args.data_dir)
    for recid in kaldi_obj.wavs:
        data, rate = kaldi_obj.load_wav(recid)
        Y = feature.stft(data, args.frame_size, args.frame_shift)
        Y = feature.transform(Y, transform_type=args.input_transform)
        Y = feature.splice(Y, context_size=args.context_size)
        Y = Y[::args.subsampling]
        out_chunks = []
        with torch.no_grad():
            hs = None
            for start, end in _gen_chunk_indices(len(Y), args.chunk_size):
                Y_chunked = torch.from_numpy(Y[start:end])
                Y_chunked.to(device)
                ys = model([Y_chunked], activation=torch.sigmoid)
                out_chunks.append(ys[0].cpu().detach().numpy())
                if args.save_attention_weight == 1:
                    raise NotImplementedError()
        outfname = recid + '.h5'
        outpath = os.path.join(args.out_dir, outfname)
        if args.label_delay != 0:
            outdata = shift(np.vstack(out_chunks), (-args.label_delay, 0))
        else:
            outdata = np.vstack(out_chunks)
        with h5py.File(outpath, 'w') as wf:
            wf.create_dataset('T_hat', data=outdata)

