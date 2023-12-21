#!/bin/bash

export CUDA_VISIBLE_DEVICES=1,7
dt=`date '+%Y%m%d_%H%M%S'`


dataset="obqa"
model='roberta-large' # '~/sdb/acenet/Pretrained/roberta-large' 
shift $(( $# > 0 ? 1 : 0 ))
shift $(( $# > 0 ? 1 : 0 ))
args=$@


elr="1e-5"
dlr="1e-3"
bs=128 #128
mbs=1
n_epochs=100
num_relation=38 #(17 +2) * 2: originally 17, add 2 relation types (QA context -> Q node; QA context -> A node), and double because we add reverse edges


k=5 #num of gnn layers
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

###### Training ######
for seed in 0; do
  python3 -u acenet.py --dataset $dataset \
      --encoder $model -k $k --gnn_dim $gnndim -elr $elr -dlr $dlr -bs $bs -mbs $mbs --fp16 true --seed $seed \
      --num_relation $num_relation \
      --n_epochs $n_epochs --max_epochs_before_stop 50  \
      --train_adj data/${dataset}/graph/train.graph.adj.pk \
      --dev_adj   data/${dataset}/graph/dev.graph.adj.pk \
      --test_adj  data/${dataset}/graph/test.graph.adj.pk \
      --train_statements data/${dataset}/statement/train-fact.statement.jsonl \
      --dev_statements   data/${dataset}/statement/dev-fact.statement.jsonl \
      --test_statements  data/${dataset}/statement/test-fact.statement.jsonl \
      --save_model \
      --save_dir ${save_dir_pref}/${dataset}/enc-${model}__k${k}__gnndim${gnndim}__bs${bs}__seed${seed}__${dt} $args \
  > logs/${dataset}__k${k}__gnndim${gnndim}__bs${bs}__seed${seed}__${dt}.log.txt
done
