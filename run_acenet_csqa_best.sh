#!/bin/bash

export CUDA_VISIBLE_DEVICES=2,1
dt=`date '+%Y%m%d-%H%M%S'`
# 目前是没有保存模型文件的

dataset="csqa"
model='roberta-large'
shift $(( $# >0 ? 1 : 0))
shift $(( $# >0 ? 1 : 0))
args=$@


elr="1e-5"
dlr="1e-3"

mbs=2
n_epochs=30
num_relation=38 #(17 +2) * 2: originally 17, add 2 relation types (QA context -> Q node; QA context -> A node), and double because we add reverse edges


k=5 #num of gnn layers
bs=64 #128的效果不是很好 可能還需要調整lr等參數
subsample=1 # subsample coefficient
gnndim=200

echo "***** hyperparameters *****"
echo "dataset: $dataset"
echo "enc_name: $model"
echo "batch_size: $bs"
echo "learning_rate: elr $elr dlr $dlr"
echo "gnn: dim $gnndim layer $k"
echo "******************************"

save_dir_pref='saved_models'
mkdir -p $save_dir_pref
mkdir -p logs
# seed=888
###### Training ######
for seed in 888; do
  python3 -u acenet_best.py --dataset $dataset \
      --encoder $model -k $k --gnn_dim $gnndim --subsample $subsample -elr $elr -dlr $dlr -bs $bs -mbs $mbs --fp16 false --seed $seed \
      --num_relation $num_relation \
      --n_epochs $n_epochs --max_epochs_before_stop 10  \
      --train_adj data/${dataset}/graph/train.graph.adj.pk \
      --dev_adj   data/${dataset}/graph/dev.graph.adj.pk \
      --test_adj  data/${dataset}/graph/test.graph.adj.pk \
      --train_statements  data/${dataset}/statement/train.statement.jsonl \
      --dev_statements  data/${dataset}/statement/dev.statement.jsonl \
      --test_statements  data/${dataset}/statement/test.statement.jsonl \
      --save_model \
      --save_dir ${save_dir_pref}/${dataset}/enc-${model}__k${k}__gnndim${gnndim}__bs${bs}__seed${seed}__${dt} \
  > logs/${dataset}__${model}__k${k}__seed${seed}__${dt}.log.txt
done
