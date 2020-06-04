#!/bin/bash
cd /ceph/qbkg2/aitingliu/MRC/bert
BASE_DIR=/ceph/qbkg2/aitingliu/MRC/chinese_roberta_wwm_ext_L-12_H-768_A-12
TRAINED_CLASSIFIER=/ceph/qbkg2/aitingliu/MRC/output/9_data_merge/model.ckpt-7302
EXPORT_DIR=/ceph/qbkg2/aitingliu/MRC/output/9_data_merge/
export CUDA_VISIBLE_DEVICES=-1

/data/anaconda3/bin/python run_squad_with_tf_record.py \
  --do_export=True \
  --do_predict=False \
  --export_version=20200525 \
  --vocab_file=$BASE_DIR/vocab.txt \
  --bert_config_file=$BASE_DIR/config.json \
  --init_checkpoint=$TRAINED_CLASSIFIER \
  --max_seq_length=256 \
  --export_dir=$EXPORT_DIR \
  --output_dir=/ceph/qbkg2/aitingliu/MRC/output/9_data_merge
