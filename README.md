# EEND_PyTorch
A PyTorch implementation of [End-to-End Neural Diarization](https://ieeexplore.ieee.org/document/9003959).

This repo is largely based on the original chainer implementation [EEND](https://github.com/hitachi-speech/EEND) by [Hitachi Ltd.](https://github.com/hitachi-speech), who holds the copyright.

This repo only includes the training/inferring part. If you are looking for data preparation, please refer to the [original authors' repo](https://github.com/hitachi-speech/EEND/blob/master/egs/callhome/v1/run_prepare_shared.sh).

## Note
Only Transformer model with PIT loss is implemented here. And I can only assure the main pipeline is correct. Some side stuffs (such as save_attn_weight, BLSTM model, deep clustering loss, etc.) are either not implemented correctly or not implemented.

Actually the orignal chainer code reserves the pytorch interface, I may consider make a merge request after the code is well-polished.

## Run
1. Prepare your kaldi-style data and modify `run.sh` according to your own directories.
2. Check configuration file. The default `conf/large/train.yaml` configuration uses a 4 layer Transformer with 100k warmsteps, which is different from their paper in ASRU2019. This configuration comes from [their paper submitted to TASLP](https://arxiv.org/abs/2003.02966). As larger model yeilds better performance.
3. `./run.sh`

## Pretrained Models
Pretrained models are offerred here.

`model_simu.th` is trained on simulation data (beta=2), and `model_callhome.th` is adapted on callhome data. They are all 4-layer Transformer models trained with `conf/large/train.yaml`.

## Results
We miss the SwitchBoard Phase 1 for training data, so the results can be a little worse.
| Type | Transformer Layer | Noam Warmup Steps | DER on simu | DER on callhome |
|:-:|:-:|:-:|:-:|:-:|
| [Chainer (ASRU2019)](https://ieeexplore.ieee.org/document/9003959) | 2 | 25k | 7.36 | 12.50 |
| [Chainer (TASLP)](https://arxiv.org/pdf/2003.02966.pdf) | 4 | 100k | 4.56 | 9.54 |
| Chainer (run on our data) | 2 | 25k | 9.78 | 14.85 |
| PyTorch (epoch 50 on simu) | 2 | 25k | 10.14 | 15.72 |
| PyTorch | 4 | 100k | 6.76 | 11.21 |
| PyTorch\* | 4 | 100k | - | 9.35 |

(\* run on full training data, credit to my great colleague!)

## Citation
Cite their great papers!
```
@inproceedings={fujita2019endtoend2,
    title={End-to-End Neural Speaker Diarization with Permutation-Free Objectives},
    author={Fujita, Yusuke and Kanda, Naoyuki and Horiguchi, Shota and Nagamatsu, Kenji and Watanabe, Shinji},
    booktitle={INTERSPEECH},
    year={2019},
    pages={4300--4304},
}
```
```
@inproceedings={fujita2019endtoend,
    title={End-to-End Neural Speaker Diarization with Self-Attention},
    author={Fujita, Yusuke and Kanda, Naoyuki and Horiguchi, Shota and Xue, Yawen and Nagamatsu, Kenji and Watanabe, Shinji},
    booktitle={IEEE Automatic Speech Recognition and Understanding Workshop (ASRU)},
    pages={296--303},
    year={2019},
}
```
```
@article={fujita2020endtoend,
    title={End-to-End Neural Diarization: Reformulating Speaker Diarization as Simple Multi-label Classification},
    author={Fujita, Yusuke and Watanabe, Shinji and Horiguchi, Shota and Xue, Yawen and Nagamatsu, Kenji},
    journal={arXiv:2003.02966},
    year={2020},
}
```
