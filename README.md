# BERT for QP-match & MRC

所有代码基于原生bert（https://github.com/google-research/bert）

预训练模型采用chinese_roberta_wwm_ext_L-12_H-768_A-12（https://github.com/ymcui/Chinese-BERT-wwm）



## QP-match

https://github.com/brightmart/roberta_zh中的`run_classifier.py`代码，做了一些不影响原始训练逻辑的改动



1. 运行`run_classifier.py`开始train/eval/predict

```shell
cd /ceph/qbkg/aitingliu/SemSim/roberta_zh/
BASE_DIR=/ceph/qbkg/aitingliu/MRC/chinese_roberta_wwm_ext_L-12_H-768_A-12
DATA_DIR=/ceph/qbkg/aitingliu/SemSim/roberta_zh/data/data_merge/qp_style_data
OUTPUT_DIR=/ceph/qbkg/aitingliu/SemSim/roberta_zh/output/11_data_merge
/data/anaconda3/bin/python run_classifier.py \
 --task_name=qp_match \
 --do_train=false \
 --do_eval=false \
 --do_predict=true \
 --data_dir=$DATA_DIR \
 --vocab_file=$BASE_DIR/vocab.txt \
 --bert_config_file=$BASE_DIR/config.json \
 --init_checkpoint=$BASE_DIR/model.ckpt \
 --max_seq_length=256 \
 --train_batch_size=32 \
 --learning_rate=2e-5 \
 --num_train_epochs=2.0 \
 --output_dir=$OUTPUT_DIR
```



2. ckpt模型生成pb模型（用于模型上线，可选）：`qp_export.sh`

```shell
#!/bin/bash
cd /ceph/qbkg2/aitingliu/SemSim/roberta_zh/
BASE_DIR=/ceph/qbkg2/aitingliu/MRC/chinese_roberta_wwm_ext_L-12_H-768_A-12
TRAINED_CLASSIFIER=/ceph/qbkg2/aitingliu/SemSim/roberta_zh/output/11_data_merge/model.ckpt-19416
EXPORT_DIR=/ceph/qbkg2/aitingliu/SemSim/roberta_zh/output/11_data_merge
export CUDA_VISIBLE_DEVICES=-1

/data/anaconda3/bin/python run_qpmatch.py \
  --task_name=qp_match \
  --do_export=True \
  --do_predict=False \
  --export_version=20200525 \
  --vocab_file=$BASE_DIR/vocab.txt \
  --bert_config_file=$BASE_DIR/config.json \
  --init_checkpoint=$TRAINED_CLASSIFIER \
  --max_seq_length=256 \
  --export_dir=$EXPORT_DIR \
  --output_dir=/ceph/qbkg2/aitingliu/MRC/output/11_data_merge \
  --data_dir=/ceph/qbkg2/aitingliu/MRC/data/data_merge
```



## MRC

bert原生`run_squad.py`代码，

因为在venus平台上45minGPU使用率为0的任务会被kill掉，当训练数据量很大时不能正常运行，

所以拆分为（训练部分逻辑不变）：

`gen_tf_record.py`：训练数据转存为.tf_record文件

`run_squad_with_tf_record.py`：`run_squad.py`省略了训练数据转tf_record的过程，直接读tf_record文件



**TODO**

另外，测试数据量很大时，没办法通过先生成tf_record文件解决，

因为写预测结果的函数write_predictions需要用到生成tf_record过程中的一个中间产物eval_examples，

后续看下这里的代码如何改动



1. 首先调用`gen_tf_record.py`生成训练数据的tf_record文件

```shell
cd "/ceph/qbkg/aitingliu/MRC/bert"
export BERT_BASE_DIR=/ceph/qbkg/aitingliu/MRC/chinese_roberta_wwm_ext_L-12_H-768_A-12
export SQUAD_DIR=/ceph/qbkg/aitingliu/MRC/data/data_merge/squad_style_data
/data/anaconda3/bin/python gen_tf_record.py \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --bert_config_file=$BERT_BASE_DIR/config.json \
  --init_checkpoint=$BERT_BASE_DIR/model.ckpt \
  --do_train=True \
  --train_file=$SQUAD_DIR/merge_squad_v2_train_qf_add_neg.json \
  --max_seq_length=256 \
  --doc_stride=128 \
  --seed=12345 \
  --version_2_with_negative=True
```

2. 调用`run_squad_with_tf_record.py`开始train/predict

```shell
export BERT_BASE_DIR=/ceph/qbkg/aitingliu/MRC/chinese_roberta_wwm_ext_L-12_H-768_A-12
export SQUAD_DIR=/ceph/qbkg/aitingliu/MRC/data/data_merge/squad_style_data
export OUTPUT_DIR=/ceph/qbkg/aitingliu/MRC/output/11_data_merge
/data/anaconda3/bin/python ./run_squad_with_tf_record.py \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --bert_config_file=$BERT_BASE_DIR/config.json \
  --init_checkpoint=$BERT_BASE_DIR/model.ckpt \
  --do_train=True \
  --train_file=$SQUAD_DIR/merge_squad_v2_train_qf_add_query_type.json \
  --do_predict=True \
  --predict_file=$SQUAD_DIR/merge_squad_v2_dev_qf_add_query_type.json \
  --train_batch_size=32 \
  --learning_rate=3e-5 \
  --num_train_epochs=2.0 \
  --max_seq_length=256 \
  --max_answer_length=30 \
  --doc_stride=128 \
  --seed=12345 \
  --output_dir=$OUTPUT_DIR \
  --version_2_with_negative=True
```

3. ckpt模型生成pb模型（用于模型上线，可选）：`squad_export.sh`

```shell
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
```

## MRC（增加问题类型作为特征）

`gen_tf_record_add_query_type.py`：对应`gen_tf_record.py`，增加问题类型字段的处ß理

`run_squad_with_tf_record_add_query_type.py`：对应`run_squad_with_tf_record.py`，增加问题类型字段的处理

增加问题类型作为特征的两种方式：

1. 在原来模型基础上，拼接问题类型作为一个特征，直接拼在【CLS】query【SEP】doc【SEP】后面，变为【CLS】query【SEP】doc【SEP】type【SEP】。BERT 的词汇表里面前 100 个预留了 99 个 [unused]，把词汇表里面的 [unused] 替换成想要的token或者在代码里做映射。
2. 不加额外的 type 和【SEP】，在原来基础上，在 modeling.py 改加一个 embedding 矩阵，问题类型 9 种维度的话，embedding 矩阵 9 种维度。（geyingli）

目前采用的是第一种方式。

1. 首先调用`gen_tf_record_add_query_type.py`生成训练数据的tf_record文件

```shell
cd "/ceph/qbkg/aitingliu/MRC/bert"
export BERT_BASE_DIR=/ceph/qbkg/aitingliu/MRC/chinese_roberta_wwm_ext_L-12_H-768_A-12
export CHECKPOINT_DIR=/ceph/qbkg/aitingliu/MRC/output/13_data_merge
export SQUAD_DIR=/ceph/qbkg/aitingliu/MRC/data/qiehao
/data/anaconda3/bin/python gen_tf_record_add_query_type.py \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --bert_config_file=$BERT_BASE_DIR/config.json \
  --init_checkpoint=$CHECKPOINT_DIR/model.ckpt-7302 \
  --do_train=True \
  --train_file=$SQUAD_DIR/qiehao_test2_add_query_type_squad_v2.json \
  --max_seq_length=256 \
  --doc_stride=128 \
  --seed=12345 \
  --version_2_with_negative=True
```



2. 调用`run_squad_with_tf_record_add_query_type.py`开始train/predict

```shell
cd "/ceph/qbkg/aitingliu/MRC/bert"
export BERT_BASE_DIR=/ceph/qbkg/aitingliu/MRC/chinese_roberta_wwm_ext_L-12_H-768_A-12
export SQUAD_DIR=/ceph/qbkg/aitingliu/MRC/data/data_merge/squad_style_data
export OUTPUT_DIR=/ceph/qbkg/aitingliu/MRC/output/13_data_merge
/data/anaconda3/bin/python ./run_squad_with_tf_record_add_query_type.py \
--vocab_file=$BERT_BASE_DIR/vocab.txt \
--bert_config_file=$BERT_BASE_DIR/config.json \
--init_checkpoint=$BERT_BASE_DIR/model.ckpt \
--do_train=True \
--train_file=$SQUAD_DIR/merge_squad_v2_train_qf_add_query_type.json \
--do_predict=True \
--predict_file=$SQUAD_DIR/merge_squad_v2_dev_qf_add_query_type.json \
--train_batch_size=32 \
--learning_rate=3e-5 \
--num_train_epochs=2.0 \
--max_seq_length=256 \
--doc_stride=128 \
--seed=12345 \
--output_dir=$OUTPUT_DIR \
--version_2_with_negative=True
```

