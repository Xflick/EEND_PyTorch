#!/usr/bin/env python3

# Copyright 2019 Hitachi, Ltd. (author: Yusuke Fujita)
# Modified by: Yexin Yang
# Licensed under the MIT license.
#
# averaging serialized models

import torch
from collections import OrderedDict
import argparse


def average_model(ifiles, ofile):
    omodel = OrderedDict()

    for ifile in ifiles:
        tmpmodel = torch.load(ifile)
        for k, v in tmpmodel.items():
            omodel[k] = omodel.get(k, 0) + v

    for k, v in omodel.items():
        omodel[k] = v / len(ifiles)

    torch.save(omodel, ofile)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("ofile")
    parser.add_argument("ifiles", nargs='+')
    args = parser.parse_args()

    print(str(args))
    average_model(args.ifiles, args.ofile)
    print("Finished averaging")

