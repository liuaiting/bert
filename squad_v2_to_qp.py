# -*- coding=utf-8 -*-
import json
import pandas as pd
from sklearn import metrics
import logging
import csv

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger()

__author__ = "aitingliu@tencent.com"


def win_n_char(c, window, stride):
  assert window >= stride
  if not stride:
    stride = window
  if len(c) <= window:
    res = [c]
  else:
    num = (len(c) - window) // stride + 2
    res = []
    for i in range(0, num):
      start = i * stride
      end = i * stride + window
      res.append(c[start:end] if end < len(c) else c[start:])
  return res


def qiefen(c, window=200, stride=50):
  paras = []
  paras.extend(win_n_char(c, window, stride))
  return paras


def gen_qiefen_qp_from_squad_v2(squad_v2_tmp_file, qp_file):
  logger.info("***** Running gen_qiefen_qp_from_squad_v2 *****")
  logger.info("squad_v2_tmp_file: {}".format(squad_v2_tmp_file))
  logger.info("qp_file: {}".format(qp_file))

  f = json.load(open(squad_v2_tmp_file))

  with open(qp_file, "w") as fw:
    csv_writer = csv.writer(fw, delimiter="\t")
    csv_writer.writerow(["label", "query", "para", "qid"])
    for data in f["data"]:
      for para in data["paragraphs"]:
        context = para["context"]
        for qas in para["qas"]:
          question = qas["question"]
          qid = qas["id"]
          contexts = qiefen(context)
          for uid, text in enumerate(contexts):
            if not qas["is_impossible"]:
              answer = qas["answers"][0]["text"]
              label = "1" if text.find(answer) != -1 else "0"
              row = [label, question, text, qid + "_" + str(uid)]
              csv_writer.writerow(row)
            else:
              row = ["0", question, text, qid + "_" + str(uid)]
              csv_writer.writerow(row)

  logger.info("***** Done gen_qiefen_qp_from_squad_v2 *****")


def qp_doc_metric(qp_file, qp_res_file, qp_doc_res_file, threshold):
  """
      Query-DOC级别指标.
      qp_file         : `label query para  qid`
      qp_res_file     : `p0  p1`
      qp_doc_res_file : `para_id  label	pred	p1`
  """
  logger.info("***** Running qp_doc_metric *****")
  logger.info("qp_file: {}".format(qp_file))
  logger.info("qp_res_file: {}".format(qp_res_file))
  logger.info("qp_doc_res_file: {}".format(qp_doc_res_file))
  logger.info("threshold: {}".format(threshold))
  input_df = pd.read_csv(qp_file, sep='\t', encoding="utf-8", header=0)
  pred_df = pd.read_csv(qp_res_file, sep='\t', encoding="utf-8", header=None, names=["p0", "p1"])
  df = pd.concat([input_df, pred_df], axis=1)
  print(df.tail())
  print(df.dtypes)

  df["para_id"] = df["qid"].map(lambda x: "_".join(x.split("_")[:2]))
  print(df.tail())

  df.loc[df.p1 > threshold, 'pred'] = 1
  df.loc[df.p1 <= threshold, 'pred'] = 0
  df["pred"] = df["pred"].astype("int")

  g_df_label = df.groupby("para_id")["label"].max()
  g_df_pred = df.groupby("para_id")["pred"].max()
  g_df_p1 = df.groupby("para_id")["p1"].max()
  g_df = pd.concat([g_df_label, g_df_pred, g_df_p1], axis=1)
  print(g_df.tail())

  print(qp_file, threshold)
  print("Query-DOC级别结果指标")
  print("THRESHOLD={} TOTAL={}".format(threshold, len(g_df_label) - 1))
  print("ACC : %.4f" % metrics.accuracy_score(g_df_label, g_df_pred))
  print("PRE : %.4f" % metrics.precision_score(g_df_label, g_df_pred))
  print("REC : %.4f" % metrics.recall_score(g_df_label, g_df_pred))
  print("F1  : %.4f" % metrics.f1_score(g_df_label, g_df_pred))
  print("AUC : %.4f" % metrics.roc_auc_score(g_df_label, g_df_p1))
  print("\n")
  g_df.to_csv(qp_doc_res_file, sep="\t", header=True, index=True, encoding="utf-8")
  logger.info("***** Done qp_doc_metric *****")


def qp_pair_metric(qp_file, qp_res_file, qp_pair_res_file, threshold):
  """
    pair级别指标.
    qp_file           : `label  query para  qid`
    qp_res_file       : `0_prob 1_prob`
    qp_pair_res_file  : `para_id  qid label pred  p1  query para`
  """
  logger.info("***** Running qp_pair_metric *****")
  logger.info("qp_file: {}".format(qp_file))
  logger.info("qp_res_file: {}".format(qp_res_file))
  logger.info("qp_pair_res_file: {}".format(qp_pair_res_file))
  logger.info("threshold: {}".format(threshold))
  input_df = pd.read_csv(qp_file, sep='\t', encoding="utf-8", header=0)
  pred_df = pd.read_csv(qp_res_file, sep='\t', encoding="utf-8", header=None, names=["p0", "p1"])
  df = pd.concat([input_df, pred_df], axis=1)
  # print(input_df.tail())
  # print(pred_df.tail())
  # print(df.tail())

  df.loc[df.p1 > threshold, 'pred'] = 1
  df.loc[df.p1 <= threshold, 'pred'] = 0
  df["pred"] = df["pred"].astype("int")

  def get_para_id(qid):
    para_id = "_".join(qid.split("_")[:2])
    return para_id

  df["para_id"] = df.qid.map(lambda x: get_para_id(x))

  print(qp_file, qp_res_file, threshold)
  print("Pair级别结果指标")
  print("THRESHOLD={} TOTAL={}".format(threshold, len(df) - 1))
  print("ACC : %.4f" % metrics.accuracy_score(df.label, df.pred))
  print("PRE : %.4f" % metrics.precision_score(df.label, df.pred))
  print("REC : %.4f" % metrics.recall_score(df.label, df.pred))
  print("F1  : %.4f" % metrics.f1_score(df.label, df.pred))
  print("AUC : %.4f" % metrics.roc_auc_score(df.label, df.p0))

  print("\n")
  df.to_csv(qp_pair_res_file, sep="\t", index=False, encoding="utf-8",
            columns=["para_id", "qid", "label", "pred", "p1", "query", "para"])
  logger.info("***** Done qp_pair_metric *****")

