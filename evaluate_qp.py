#!/usr/bin/env python
# -*- coding:utf-8 -*-
import os
import pandas as pd
from sklearn import metrics
import matplotlib.pyplot as plt
import numpy as np
import csv

__author__ = "aitingliu@tencent.com"


def metric(input_path, predict_path, threshold):
    """
    Query-Para级别指标.
    [test data format]: label\tquery\tpara\tqid
    [prediction file format]: 0_prob\t1_prob
    :param input_path:
    :param threshold:
    :return:
    """
    ############ winsechang : csv读文件 #########
    # with open(input_path) as f:
    #     labels = []
    #     querys = []
    #     paras = []
    #     qids = []
    #     f.readline()
    #     for line in f:
    #         line = line.strip().split("\t")
    #         labels.append(int(line[0]))
    #         querys.append(line[1])
    #         paras.append(line[2])
    #         qids.append(line[3])
    # with open(predict_path) as f:
    #     p1 = []
    #     for line in f:
    # #         p1.append(float(line.strip().split("\t")[1]))
    # df = pd.DataFrame({"query": querys, "para": paras, "label": labels, "p1": p1, "qid": qids})

    ###############  update(aitingliu): pandas读文件   #############
    input_df = pd.read_csv(input_path, sep='\t', encoding="utf-8", header=0)
    pred_df = pd.read_csv(predict_path, sep='\t', encoding="utf-8", header=None, names=["p0", "p1"])
    df = pd.concat([input_df, pred_df], axis=1)

    ########### 根据预测概率分布（预测为0或1的概率），卡阈值做0/1预测   #########
    df.loc[df.p1 > threshold, 'pred'] = 1
    df.loc[df.p1 <= threshold, 'pred'] = 0
    df["pred"] = df["pred"].astype("int")

    ######### Pair级别各项指标 #########
    print(input_path, predict_path, threshold)
    print("Query-Para级别结果指标")
    print("THRESHOLD={} TOTAL={}".format(threshold, len(df) - 1))
    print("ACC : %.4f" % metrics.accuracy_score(df.label, df.pred))
    print("PRE : %.4f" % metrics.precision_score(df.label, df.pred))
    print("REC : %.4f" % metrics.recall_score(df.label, df.pred))
    print("F1  : %.4f" % metrics.f1_score(df.label, df.pred))
    print("AUC : %.4f" % metrics.roc_auc_score(df.label, df.p1))
    print("\n")
    df.to_csv("{}_pred".format(predict_path), sep="\t", index=False, encoding="utf-8",
              columns=["qid", "label", "pred", "p1", "query", "para"])

    ################ Precision-Recall Curve ##############
    plt.figure()
    plt.title('Precision/Recall Curve')  # give plot a title
    plt.xlabel('Recall')  # make axis labels
    plt.ylabel('Precision')
    # y_true和y_scores分别是gt label和predict score
    y_true = df.label
    y_scores = df.p1
    precision, recall, thresholds = metrics.precision_recall_curve(y_true, y_scores)
    plt.plot(precision, recall)
    # plt.show()
    plt.savefig("{}_pr.png".format(input_path))
    plt.close()

    ################ ROC Curve ##############
    y_true = df.label
    y_score = df.p1
    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_score)
    # print('fpr: ' + str(fpr))
    # print('tpr: ' + str(tpr))
    # print('thresholds: ' + str(thresholds))

    AUC = metrics.auc(fpr, tpr)
    # print('AUC: ' + str(AUC))

    print('\n\n####################################################')
    plt.figure()
    plt.plot(fpr, tpr)
    plt.title('ROC Curve' + '(AUC: ' + str(AUC) + ')')
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.savefig("{}_roc.png".format(input_path))
    plt.close()

    ################ Confusion Matrix ##############
    cm = metrics.confusion_matrix(df.label, df.pred)
    plt.figure()
    plt.matshow(cm, cmap=plt.cm.Greens)
    plt.title("Confusion Matrix")

    # 画混淆矩阵图，配色风格使用cm.Greens
    plt.colorbar()

    # 颜色标签
    for x in range(len(cm)):
        for y in range(len(cm)):
            plt.annotate(cm[x, y], xy=(x, y), horizontalalignment='center', verticalalignment='center')

    plt.ylabel('True label')  # 坐标轴标签
    plt.xlabel('Predicted label')  # 坐标轴标签

    plt.savefig("{}_cm.png".format(input_path))
    plt.close()


# def total_metric(input_path, output_path, threshold):
#     """
#     [deprecated]: 针对QQ匹配的指标
#     Query级别指标.
#     [input_format]: qid\tlabel\tpred\tp1\tquery\tpara
#     :param input_path:
#     :param output_path:
#     :param threshold:
#     :return:
#     """
#     output_dir = os.path.dirname(output_path)
#     if not os.path.exists(output_dir):
#         os.makedirs(output_dir)
#
#     d = {}
#     for line in open(input_path, "r", encoding="utf-8"):
#         qid, label, predict, p1, query, para = line.strip().split("\t")
#         if query == "query":
#             continue
#         d.setdefault(query, [])
#         d[query].append((para, label, predict))
#
#     labels = []
#     scores = []
#     pred_labels = []
#
#     f = open(output_path, "w", encoding="utf-8")
#     P, N, TP, FN, FP, TN = 0, 0, 0, 0, 0, 0
#     for para_id in d:
#         terms = d[para_id]
#
#         sort_l = [(float(predict), (para, label, predict)) for para, label, predict in
#                   terms]  # 针对每一个query的所有question预测结果排序
#
#         sort_l.sort(reverse=True)
#         terms_str = "\t".join(["-".join(list(term[1])) for term in sort_l])
#         # terms_str = str(sort_l)
#
#         best_term = sort_l[0][1]
#         q2, label, predict = best_term
#         # print(best_term)
#
#         predict_label = 1 if float(predict) > threshold else 0  # best answer的结果 是否过阈值
#         # predict_answer = predict_label * int(label) # 判定best answer预测结果是否正确
#         has_answer = int(any([int(label) for para, label, predict in terms]))  # 判定候选question是否有正确答案
#         # print(has_answer)
#         # predict_answer has_answer  q1 q2_list 作为最终的判定序列
#         f.write("\t".join([str(has_answer), label, str(predict_label), predict, para_id, terms_str]) + "\n")
#
#         label = int(label)
#         if has_answer == 1: P += 1  # 正类 候选包含正确答案
#         if has_answer == 0: N += 1  # 负类 候选不包含正确答案
#         if label == 1 and predict_label == 1: TP += 1  # 将正类预测为正类数
#         if label == 1 and predict_label == 0: FN += 1  # 将正类预测为负类数
#         if label == 0 and predict_label == 1: FP += 1  # 将负类预测为正类数
#         if label == 0 and predict_label == 0: TN += 1  # 将负类预测为负类数
#
#         labels.append(label)
#         scores.append(predict)
#         pred_labels.append(predict_label)
#
#     PRE = TP / (TP + FP + 0.00000001)
#     REC = TP / (P + 0.00000001)
#     ACC = (TP + TN) / (P + N + 0.00000001)
#     F1 = 2 * PRE * REC / (PRE + REC + 0.00000001)
#
#     print(input_path, output_path, threshold)
#     print("Query级别结果指标")
#     print("THRESHOLD={} P={} N={} TP={} FN={} FP={} TN={}".format(threshold, P, N, TP, FN, FP, TN))
#     print("ACC : %.4f" % ACC)
#     print("PRE : %.4f" % PRE)
#     print("REC : %.4f" % REC)
#     print("F1  : %.4f" % F1)
#     print("\n")


def doc_metric(input_path, threshold):
    """
    Query-DOC级别指标.
    [input_format]: qid\tlabel\tpred\tp1\tquery\tpara
    [output_format]: para_id
    :param input_path:
    :param threshold:
    :return:
    """
    df = pd.read_csv(input_path, sep="\t", encoding="utf-8", header=0)

    def get_para_id(qid):
        para_id = "_".join(qid.split("_")[:2])
        return para_id

    df["para_id"] = df.qid.map(lambda x: get_para_id(x))

    g_df_label = df.groupby("para_id")["label"].max()
    g_df_pred = df.groupby("para_id")["pred"].max()
    g_df_p1 = df.groupby("para_id")["p1"].max()
    g_df = pd.concat([g_df_label, g_df_pred, g_df_p1], axis=1)
    # g_df_label = df.groupby("query")["label"].max()
    # g_df_pred = df.groupby("query")["pred"].max()
    # g_df_p1 = df.groupby("query")["p1"].max()
    # g_df = pd.concat([g_df_label, g_df_pred, g_df_p1], axis=1)


    print(input_path, threshold)
    print("Query-DOC级别结果指标")
    print("THRESHOLD={} TOTAL={}".format(threshold, len(g_df_label) - 1))
    print("ACC : %.4f" % metrics.accuracy_score(g_df_label, g_df_pred))
    print("PRE : %.4f" % metrics.precision_score(g_df_label, g_df_pred))
    print("REC : %.4f" % metrics.recall_score(g_df_label, g_df_pred))
    print("F1  : %.4f" % metrics.f1_score(g_df_label, g_df_pred))
    print("AUC : %.4f" % metrics.roc_auc_score(g_df_label, g_df_p1))
    print("\n")
    g_df.to_csv("{}_doc".format(input_path), sep="\t", header=True, index=True, encoding="utf-8")


if __name__ == "__main__":
    # metric("data/dureader/final/test.tsv", "output/1_dureader/test_results.tsv", 0.5)
    # metric("data/dureader/final/test.tsv", "output/2_dureader/test_results.tsv", 0.5)
    # metric("data/dureader/final/test.tsv", "output/3_dureader/test_results.tsv", 0.5)
    # metric("data/dureader/final/test.tsv", "output/4_dureader/test_results.tsv", 0.5)
    # metric("data/dureader/final/test.tsv", "output/5_dureader/test_results.tsv", 0.5)
    # metric("data/cmrc2018/qp_style_data/test.tsv", "output/6_cmrc2018/test_results.tsv", 0.5)
    # metric("data/dureader/qp_style_data/test.tsv", "output/7_dureader/test_results.tsv", 0.5)
    # metric("data/dureader/final_query_filter/test.tsv", "output/8_dureader_final_query_filter/test_results.tsv", 0.5)
    # metric("data/cmrc2018/qp_query_filter/test.tsv", "output/9_cmrc2018_qp_query_filter/test_results.tsv", 0.5)
    # metric("data/dureader/qp_query_filter/test.tsv", "output/10_dureader_qp_query_filter/test_results.tsv", 0.5)
    # metric("data/baidu_short/test.tsv", "output/9_cmrc2018_qp_query_filter/9_baidu_short_test_results.tsv", 0.5)
    # metric("data/qiehao/query_test2/test.tsv", "output/9_cmrc2018_qp_query_filter/9_qiehao_test2_test_results.tsv", 0.5)
    # metric("data/data_merge/qp_style_data/test.tsv", "output/11_data_merge/test_results.tsv", 0.5)

    metric("data_merge/qp_style_data/test.tsv", "data_merge/qp_style_data/test_results.tsv", 0.5)
    doc_metric("data_merge/qp_style_data/test_results.tsv_pred", 0.5)

    metric("qiehao/biaozhu/test1/企鹅号文章预处理后/test.tsv", "qiehao/biaozhu/test1/企鹅号文章预处理后/test_results.tsv", 0.5)
    doc_metric("qiehao/biaozhu/test1/企鹅号文章预处理后/test_results.tsv_pred", 0.5)

    metric("qiehao/biaozhu/test2/企鹅号文章预处理后/test.tsv", "qiehao/biaozhu/test2/企鹅号文章预处理后/test_results.tsv", 0.5)
    doc_metric("qiehao/biaozhu/test2/企鹅号文章预处理后/test_results.tsv_pred", 0.5)


