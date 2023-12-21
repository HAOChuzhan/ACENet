#!/bin/bash

export CUDA_VISIBLE_DEVICES=6
dt=`date '+%Y%m%d_%H%M%S'`


dataset="obqa"
model='roberta-large'
shift $(( $# >0 ? 1 : 0))
shift $(( $# >0 ? 1 : 0))
args=$@


echo "******************************"
echo "dataset: $dataset"
echo "******************************"

save_dir_pref='saved_models'
mkdir -p $save_dir_pref

###### Eval ######
python3 -u acenet_best.py --dataset $dataset \
      --train_adj data/${dataset}/graph/train.graph.adj.pk \
      --dev_adj   data/${dataset}/graph/dev.graph.adj.pk \
      --test_adj  data/${dataset}/graph/test.graph.adj.pk \
      --train_statements data/${dataset}/statement/train.statement.jsonl \
      --dev_statements   data/${dataset}/statement/dev.statement.jsonl \
      --test_statements  data/${dataset}/statement/test.statement.jsonl \
      --save_model \
      --save_dir saved_models \
      --mode eval_detail \
      --load_model_path saved_models/obqa/enc-roberta-large__k5__gnndim200__bs64__seed888__20220512_223730/model.pt.65 \
      $args
# /data1/kbqa/hcz/Projects/MHIAN/saved_models/obqa/enc-roberta-large__k5__gnndim200__bs64__seed888__20220512_223730/model.pt.65