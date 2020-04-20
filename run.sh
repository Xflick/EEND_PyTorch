#!/bin/bash

ulimit -S -n 4096

export PYTHONPATH=`pwd`:$PYTHONPATH

exp_dir=exp_large
conf_dir=conf/large

train_dir=data/simu/data/swb_sre_tr_ns2_beta2_100000
dev_dir=data/simu/data/swb_sre_cv_ns2_beta2_500
model_dir=$exp_dir/models
train_conf=$conf_dir/train.yaml

train_adapt_dir=data/eval/callhome1_spk2
dev_adapt_dir=data/eval/callhome2_spk2
model_adapt_dir=$exp_dir/models_adapt
adapt_conf=$conf_dir/adapt.yaml
init_model=$model_dir/avg.th

infer_conf=$conf_dir/infer.yaml
test_dir=data/eval/callhome2_spk2
test_model=$model_adapt_dir/avg.th
infer_out_dir=$exp_dir/infer/callhome
# test_dir=data/simu/data/swb_sre_cv_ns2_beta2_500
# test_model=$model_dir/avg.th
# infer_out_dir=$exp_dir/infer/simu

work=$infer_out_dir/.work
scoring_dir=$exp_dir/score/callhome
# scoring_dir=$exp_dir/score/simu

stage=1

# Training
if [ $stage -le 1 ]; then
    echo "Start training"
    python eend/bin/train.py -c $train_conf $train_dir $dev_dir $model_dir
fi

# Model averaging
if [ $stage -le 2 ]; then
    echo "Start model averaging"
    ifiles=`eval echo $model_dir/transformer{91..100}.th`
    python eend/bin/model_averaging.py $init_model $ifiles
fi

# Adapting
if [ $stage -le 3 ]; then
    echo "Start adapting"
    python eend/bin/train.py -c $adapt_conf $train_adapt_dir $dev_adapt_dir $model_adapt_dir --initmodel $init_model
fi

# Model averaging
if [ $stage -le 4 ]; then
    echo "Start model averaging"
    ifiles=`eval echo $model_adapt_dir/transformer{91..100}.th`
    python eend/bin/model_averaging.py $test_model $ifiles
fi

# Inferring
if [ $stage -le 5 ]; then
    echo "Start inferring"
    python eend/bin/infer.py -c $infer_conf $test_dir $test_model $infer_out_dir
fi

# Scoring
if [ $stage -le 6 ]; then
    echo "Start scoring"
    mkdir -p $work
    mkdir -p $scoring_dir
	find $infer_out_dir -iname "*.h5" > $work/file_list
	for med in 1 11; do
	for th in 0.3 0.4 0.5 0.6 0.7; do
	python eend/bin/make_rttm.py --median=$med --threshold=$th \
		--frame_shift=80 --subsampling=10 --sampling_rate=8000 \
		$work/file_list $scoring_dir/hyp_${th}_$med.rttm
	md-eval.pl -c 0.25 \
		-r $test_dir/rttm \
		-s $scoring_dir/hyp_${th}_$med.rttm > $scoring_dir/result_th${th}_med${med}_collar0.25 2>/dev/null || exit
	done
	done
fi
