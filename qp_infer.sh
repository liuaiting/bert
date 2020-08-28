#!/bin/bash
cd /ceph/qbkg/aitingliu/DeepQA_v1/bert/
BASE_DIR=/ceph/qbkg/aitingliu/DeepQA_v1/pretrained_model/chinese_roberta_wwm_ext_L-12_H-768_A-12
DATA_DIR=/ceph/qbkg/aitingliu/DeepQA_v1/qp_data/baidubaike/monthquery/query1kw_0000
OUTPUT_DIR=/ceph/qbkg/aitingliu/DeepQA_v1/qp_output/11_data_merge


for i in {0..9}
do
    FILE_NAME=`echo $i | awk '{printf("query1w_%04d\n",$0)}'`
    /data/anaconda3/bin/python run_classifier.py \
        --task_name=qp_pair \
        --do_train=false \
        --do_eval=false \
        --do_predict=true \
        --data_dir=$DATA_DIR/$FILE_NAME \
        --vocab_file=$BASE_DIR/vocab.txt \
        --bert_config_file=$BASE_DIR/config.json \
        --init_checkpoint=$BASE_DIR/model.ckpt \
        --max_seq_length=256 \
        --train_batch_size=32 \
        --learning_rate=2e-5 \
        --num_train_epochs=2.0 \
        --output_dir=$OUTPUT_DIR
done
