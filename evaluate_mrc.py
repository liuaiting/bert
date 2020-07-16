"""Official evaluation script for SQuAD version 2.0.

In addition to basic functionality, we also compute additional statistics and
plot precision-recall curves if an additional na_prob.json file is provided.
This file is expected to map question ID's to the model's predicted probability
that a question is unanswerable.
"""
# update 20200716(aitingliu): 在squad2.0评测脚本的基础上进行改动
import argparse
import collections
import json
import numpy as np
import os
import re
import string
import sys

# from zhon import hanzi
# import nltk


OPTS = None


def parse_args():
  parser = argparse.ArgumentParser('Official evaluation script for SQuAD version 2.0.')
  parser.add_argument('data_file', metavar='data.json', help='Input data JSON file.')
  parser.add_argument('pred_file', metavar='pred.json', help='Model predictions.')
  parser.add_argument('--out-file', '-o', metavar='eval.json',
                      help='Write accuracy metrics to file (default is stdout).')
  parser.add_argument('--na-prob-file', '-n', metavar='na_prob.json',
                      help='Model estimates of probability of no answer.')
  parser.add_argument('--na-prob-thresh', '-t', type=float, default=1.0,
                      help='Predict "" if no-answer probability exceeds this (default = 1.0).')
  parser.add_argument('--out-image-dir', '-p', metavar='out_images', default=None,
                      help='Save precision-recall curves to directory.')
  parser.add_argument('--verbose', '-v', action='store_true')
  if len(sys.argv) == 1:
    parser.print_help()
    sys.exit(1)
  return parser.parse_args()


def make_qid_to_has_ans(dataset):
  qid_to_has_ans = {}
  for article in dataset:
    for p in article['paragraphs']:
      for qa in p['qas']:
        qid_to_has_ans[qa['id']] = bool(qa['answers'])
  return qid_to_has_ans


def normalize_answer(s):
  """Lower text and remove punctuation, articles and extra whitespace."""

  # def remove_articles(text):
  #     regex = re.compile(r'\b(a|an|the)\b', re.UNICODE)
  #     return re.sub(regex, ' ', text)

  def white_space_fix(text):
    return ' '.join(text.split())

  def remove_punc(text):
    # sp_char = [
    #     u':', u'_', u'`', u'，', u'。', u'：', u'？', u'！', u'(', u')',
    #     u'“', u'”', u'；', u'’', u'《', u'》', u'……', u'·', u'、', u',',
    #     u'「', u'」', u'（', u'）', u'－', u'～', u'『', u'』', '|'
    # ]
    exclude_en = set(string.punctuation)
    # exclude_zh = set(hanzi.punctuation)
    exclude_zh = set(['〰', '＋', '＞', '｢', '？', '〖', '＃', '：', '’', '＆', '﹑', '〔', '》', '‟', '､',
                      '〕', '【', '～', '＄', '〉', '「', '『', '。', '／', '〈', '〜', '－', '〛', '）', '；',
                      '｟', '〘', '｣', '（', '—', '‛', '〟', '＾', '…', '＇', '，', '”', '％', '〚', '＊', '＂',
                      '‧', '〙', '＜', '‘', '！', '｠', '＼', '„', '】', '《', '〞', '﹔', '〃', '＿', '＝', '』',
                      '·', '\u3000', '﹏', '｝', '“', '］', '［', '–', '｜', '〿', '｀', '、', '〗', '｡', '〾', '」',
                      '｛', '〝', '＠'])
    exclude = exclude_en | exclude_zh

    return ''.join(ch for ch in text if ch not in exclude)

  def lower(text):
    return text.lower()

  res = white_space_fix(remove_punc(lower(s)))
  # print(res)
  return res


# def get_tokens(s):
#   if not s: return []
#   return normalize_answer(s).split()


def get_tokens(s):
  """
   TODO(aitingliu): 中英文切分，中文以字为单位，英文以词为单位
   Cut the sentence into the format we want:
  - continous letters and symbols like back-slash and parenthese
  - single Chineses character  - other symbols
  """
  s = normalize_answer(s)
  regex = []
  regex += [r'[0-9a-zA-Z\\+\-<>.]+']  # English and number part for type name.
  # regex += [r'[\u4e00 -\u9fa5]']       # Chinese characters part.
  regex += [
    r'[\u4E00-\u9FFF\u3400-\u4DBF\u20000-\u2A6DF\u2A700-\u2B73F\u2B740-\u2B81F\u2B820-\u2CEAF\uF900-\uFAFF\u2F800-\u2FA1F]']  # Chinese characters part.
  regex += [r'[^\s]']  # Exclude the space.
  regex = '|'.join(regex)
  _RE = re.compile(regex)
  segs = _RE.findall(s.strip())
  # print(segs)
  return segs


def compute_em(a_gold, a_pred):
  """计算一个预测答案和一个标准答案的em值（squad2.0/LIC2020/CMRC2018的评测脚本都一样）"""
  em = int(normalize_answer(a_gold) == normalize_answer(a_pred))
  # print(em)
  return em


def compute_f1(a_gold, a_pred):
  """计算一个预测答案和一个标准答案的f1值（squad2.0评测脚本）"""
  gold_toks = get_tokens(a_gold)
  pred_toks = get_tokens(a_pred)
  common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
  num_same = sum(common.values())
  if len(gold_toks) == 0 or len(pred_toks) == 0:
    # If either is no-answer, then F1 is 1 if they agree, 0 otherwise
    return int(gold_toks == pred_toks)
  if num_same == 0:
    return 0
  precision = 1.0 * num_same / len(pred_toks)
  recall = 1.0 * num_same / len(gold_toks)
  f1 = (2 * precision * recall) / (precision + recall)
  # print(f1)
  return f1


def find_lcs(s1, s2):
  """find the longest common subsequence between s1 ans s2（LIC2020和CMRC2018的评测脚本）"""
  m = [[0 for i in range(len(s2) + 1)] for j in range(len(s1) + 1)]
  max_len = 0
  p = 0
  for i in range(len(s1)):
    for j in range(len(s2)):
      if s1[i] == s2[j]:
        m[i + 1][j + 1] = m[i][j] + 1
        if m[i + 1][j + 1] > max_len:
          max_len = m[i + 1][j + 1]
          p = i + 1
  return s1[p - max_len:p], max_len


def compute_f1_lcs(a_gold, a_pred):
  """计算一个预测答案和一个标准答案的f1值（LIC2020和CMRC2018的评测脚本）"""
  ans_segs = get_tokens(a_gold)
  prediction_segs = get_tokens(a_pred)
  # if args.debug:
  #     print(json.dumps(ans_segs, ensure_ascii=False))
  #     print(json.dumps(prediction_segs, ensure_ascii=False))
  lcs, lcs_len = find_lcs(ans_segs, prediction_segs)
  if lcs_len == 0:
    f1 = 0
  else:
    prec = 1.0 * lcs_len / len(prediction_segs)
    rec = 1.0 * lcs_len / len(ans_segs)
    f1 = (2 * prec * rec) / (prec + rec)

  return f1


def get_raw_scores(dataset, preds):
  em_scores = {}
  f1_scores = {}
  f1_lcs_scores = {}
  for article in dataset:
    for p in article['paragraphs']:
      for qa in p['qas']:
        qid = qa['id']
        gold_answers = [a['text'] for a in qa['answers']
                        if normalize_answer(a['text'])]
        if not gold_answers:
          # For unanswerable questions, only correct answer is empty string
          gold_answers = ['']
        if qid not in preds:
          print('Missing prediction for %s' % qid)
          continue
        a_pred = preds[qid]
        # Take max over all gold answers
        em_scores[qid] = max(compute_em(a, a_pred) for a in gold_answers)
        f1_scores[qid] = max(compute_f1(a, a_pred) for a in gold_answers)
        f1_lcs_scores[qid] = max(compute_f1_lcs(a, a_pred) for a in gold_answers)
  return em_scores, f1_scores, f1_lcs_scores


def apply_no_ans_threshold(scores, na_probs, qid_to_has_ans, na_prob_thresh):
  new_scores = {}
  for qid, s in scores.items():
    pred_na = na_probs[qid] > na_prob_thresh
    if pred_na:
      new_scores[qid] = float(not qid_to_has_ans[qid])
    else:
      new_scores[qid] = s
  return new_scores


def make_eval_dict(em_scores, f1_scores, f1_lcs_scores, qid_list=None):
  if not qid_list:
    total = len(em_scores)
    return collections.OrderedDict([
      ('em', 100.0 * sum(em_scores.values()) / total),
      ('f1', 100.0 * sum(f1_scores.values()) / total),
      ('f1_lcs', 100.0 * sum(f1_lcs_scores.values()) / total),
      ('total', total),
    ])
  else:
    total = len(qid_list)
    return collections.OrderedDict([
      ('em', 100.0 * sum(em_scores[k] for k in qid_list) / total),
      ('f1', 100.0 * sum(f1_scores[k] for k in qid_list) / total),
      ('f1_lcs', 100.0 * sum(f1_lcs_scores[k] for k in qid_list) / total),
      ('total', total),
    ])


def merge_eval(main_eval, new_eval, prefix):
  for k in new_eval:
    main_eval['%s_%s' % (prefix, k)] = new_eval[k]


def plot_pr_curve(precisions, recalls, out_image, title):
  plt.step(recalls, precisions, color='b', alpha=0.2, where='post')
  plt.fill_between(recalls, precisions, step='post', alpha=0.2, color='b')
  plt.xlabel('Recall')
  plt.ylabel('Precision')
  plt.xlim([0.0, 1.05])
  plt.ylim([0.0, 1.05])
  plt.title(title)
  plt.savefig(out_image)
  plt.clf()


def make_precision_recall_eval(scores, na_probs, num_true_pos, qid_to_has_ans,
                               out_image=None, title=None):
  qid_list = sorted(na_probs, key=lambda k: na_probs[k])
  true_pos = 0.0
  cur_p = 1.0
  cur_r = 0.0
  precisions = [1.0]
  recalls = [0.0]
  avg_prec = 0.0
  for i, qid in enumerate(qid_list):
    if qid_to_has_ans[qid]:
      true_pos += scores[qid]
    cur_p = true_pos / float(i + 1)
    cur_r = true_pos / float(num_true_pos)
    if i == len(qid_list) - 1 or na_probs[qid] != na_probs[qid_list[i + 1]]:
      # i.e., if we can put a threshold after this point
      avg_prec += cur_p * (cur_r - recalls[-1])
      precisions.append(cur_p)
      recalls.append(cur_r)
  if out_image:
    plot_pr_curve(precisions, recalls, out_image, title)
  return {'ap': 100.0 * avg_prec}


def run_precision_recall_analysis(main_eval, em_raw, f1_raw, na_probs,
                                  qid_to_has_ans, out_image_dir):
  if out_image_dir and not os.path.exists(out_image_dir):
    os.makedirs(out_image_dir)
  num_true_pos = sum(1 for v in qid_to_has_ans.values() if v)
  if num_true_pos == 0:
    return
  pr_em = make_precision_recall_eval(
    em_raw, na_probs, num_true_pos, qid_to_has_ans,
    out_image=os.path.join(out_image_dir, 'pr_em.png'),
    title='Precision-Recall curve for Exact Match score')
  pr_f1 = make_precision_recall_eval(
    f1_raw, na_probs, num_true_pos, qid_to_has_ans,
    out_image=os.path.join(out_image_dir, 'pr_f1.png'),
    title='Precision-Recall curve for F1 score')
  oracle_scores = {k: float(v) for k, v in qid_to_has_ans.items()}
  pr_oracle = make_precision_recall_eval(
    oracle_scores, na_probs, num_true_pos, qid_to_has_ans,
    out_image=os.path.join(out_image_dir, 'pr_oracle.png'),
    title='Oracle Precision-Recall curve (binary task of HasAns vs. NoAns)')
  merge_eval(main_eval, pr_em, 'pr_em')
  merge_eval(main_eval, pr_f1, 'pr_f1')
  merge_eval(main_eval, pr_oracle, 'pr_oracle')


def histogram_na_prob(na_probs, qid_list, image_dir, name):
  if not qid_list:
    return
  x = [na_probs[k] for k in qid_list]
  weights = np.ones_like(x) / float(len(x))
  plt.hist(x, weights=weights, bins=20, range=(0.0, 1.0))
  plt.xlabel('Model probability of no-answer')
  plt.ylabel('Proportion of dataset')
  plt.title('Histogram of no-answer probability: %s' % name)
  plt.savefig(os.path.join(image_dir, 'na_prob_hist_%s.png' % name))
  plt.clf()


def find_best_thresh(preds, scores, na_probs, qid_to_has_ans):
  num_no_ans = sum(1 for k in qid_to_has_ans if not qid_to_has_ans[k])
  cur_score = num_no_ans
  best_score = cur_score
  best_thresh = 0.0
  qid_list = sorted(na_probs, key=lambda k: na_probs[k])
  for i, qid in enumerate(qid_list):
    if qid not in scores: continue
    if qid_to_has_ans[qid]:
      diff = scores[qid]
    else:
      if preds[qid]:
        diff = -1
      else:
        diff = 0
    cur_score += diff
    if cur_score > best_score:
      best_score = cur_score
      best_thresh = na_probs[qid]
  return 100.0 * best_score / len(scores), best_thresh


def find_all_best_thresh(main_eval, preds, em_raw, f1_raw, f1_lcs_raw, na_probs, qid_to_has_ans):
  best_em, em_thresh = find_best_thresh(preds, em_raw, na_probs, qid_to_has_ans)
  best_f1, f1_thresh = find_best_thresh(preds, f1_raw, na_probs, qid_to_has_ans)
  # best_f1_lcs, f1_lcs_thresh = find_best_thresh(preds, f1_lcs_raw, na_probs, qid_to_has_ans)
  main_eval['best_em'] = best_em
  main_eval['best_em_thresh'] = em_thresh
  main_eval['best_f1'] = best_f1
  main_eval['best_f1_thresh'] = f1_thresh
  # main_eval['best_f1_lcs'] = best_f1_lcs
  # main_eval['best_f1_lcs_thresh'] = f1_lcs_thresh


def main():
  with open(OPTS.data_file) as f:
    dataset_json = json.load(f)
    dataset = dataset_json['data']
  with open(OPTS.pred_file) as f:
    preds = json.load(f)
  if OPTS.na_prob_file:
    with open(OPTS.na_prob_file) as f:
      na_probs = json.load(f)
  else:
    na_probs = {k: 0.0 for k in preds}
  qid_to_has_ans = make_qid_to_has_ans(dataset)  # maps qid to True/False
  has_ans_qids = [k for k, v in qid_to_has_ans.items() if v]
  no_ans_qids = [k for k, v in qid_to_has_ans.items() if not v]
  em_raw, f1_raw, f1_lcs_raw = get_raw_scores(dataset, preds)
  em_thresh = apply_no_ans_threshold(em_raw, na_probs, qid_to_has_ans,
                                     OPTS.na_prob_thresh)
  f1_thresh = apply_no_ans_threshold(f1_raw, na_probs, qid_to_has_ans,
                                     OPTS.na_prob_thresh)
  f1_lcs_thresh = apply_no_ans_threshold(f1_lcs_raw, na_probs, qid_to_has_ans,
                                         OPTS.na_prob_thresh)
  out_eval = make_eval_dict(em_thresh, f1_thresh, f1_lcs_thresh)
  if has_ans_qids:
    has_ans_eval = make_eval_dict(em_thresh, f1_thresh, f1_lcs_thresh, qid_list=has_ans_qids)
    merge_eval(out_eval, has_ans_eval, 'HasAns')
  if no_ans_qids:
    no_ans_eval = make_eval_dict(em_thresh, f1_thresh, f1_lcs_thresh, qid_list=no_ans_qids)
    merge_eval(out_eval, no_ans_eval, 'NoAns')
  if OPTS.na_prob_file:
    find_all_best_thresh(out_eval, preds, em_raw, f1_raw, f1_lcs_raw, na_probs, qid_to_has_ans)
  if OPTS.na_prob_file and OPTS.out_image_dir:
    run_precision_recall_analysis(out_eval, em_raw, f1_raw, na_probs,
                                  qid_to_has_ans, OPTS.out_image_dir)
    histogram_na_prob(na_probs, has_ans_qids, OPTS.out_image_dir, 'hasAns')
    histogram_na_prob(na_probs, no_ans_qids, OPTS.out_image_dir, 'noAns')
  if OPTS.out_file:
    with open(OPTS.out_file, 'w') as f:
      json.dump(out_eval, f)
  else:
    print(json.dumps(out_eval, indent=2))


# split Chinese with English（cmrc2018评估脚本）
# def mixed_segmentation(in_str, rm_punc=False):
#     in_str = str(in_str).lower().strip()
#     segs_out = []
#     temp_str = ""
#     # sp_char = ['-', ':', '_', '*', '^', '/', '\\', '~', '`', '+', '=',
#     #            '，', '。', '：', '？', '！', '“', '”', '；', '’', '《', '》', '……', '·', '、',
#     #            '「', '」', '（', '）', '－', '～', '『', '』']
#     exclude_en = set(string.punctuation)
#     # exclude_zh = set(hanzi.punctuation)
#     exclude_zh = set({'〰', '＋', '＞', '｢', '？', '〖', '＃', '：', '’', '＆', '﹑', '〔', '》', '‟', '､', '〕', '【', '～', '＄', '〉', '「', '『', '。', '／', '〈', '〜', '－', '〛', '）', '；', '｟', '〘', '｣', '（', '—', '‛', '〟', '＾', '…', '＇', '，', '”', '％', '〚', '＊', '＂', '‧', '〙', '＜', '‘', '！', '｠', '＼', '„', '】', '《', '〞', '﹔', '〃', '＿', '＝', '』', '·', '\u3000', '﹏', '｝', '“', '］', '［', '–', '｜', '〿', '｀', '、', '〗', '｡', '〾', '」', '｛', '〝', '＠'})
#     exclude = exclude_en | exclude_zh
#     sp_char = list(exclude)

#     for char in in_str:
#         if rm_punc and char in sp_char:
#             continue
#         if re.search('[\u4e00-\u9fa5]', char) or char in sp_char:
#             if temp_str != "":
#                 ss = nltk.word_tokenize(temp_str)
#                 segs_out.extend(ss)
#                 print(segs_out)
#                 temp_str = ""
#             segs_out.append(char)
#         else:
#             temp_str += char

#     # handling last part
#     if temp_str != "":
#         ss = nltk.word_tokenize(temp_str)
#         segs_out.extend(ss)

#     return segs_out


def test():
  a = """你是   谁  。；fdfjifd hufhd Fijifd 什么Fifd               ifd jfidjf     fdifdF是呢  粉底 ！！~！@#@#~！@#￥%……&*（）——+「」「|：“《》？·-=【】、；‘，。、~!@#$%^&*()_+$￥|:"<>?`[]',./"""
  b = """df 你你是   谁  。；fdfjifd 你hufhd Fijifd 什么Fifd               ifd jfidjf     fdifdF是呢  粉底 ！！~！@#@#~！@#￥%……&*（）——+「」「|：“《》？·-=【】、；‘，。、~!@#$%^&*()_+$￥|:"<>?`[]',./"""
  normalize_answer(a)
  print(get_tokens(a))
  a = """你是   谁  。；fdfjifd hufhd Fijifd 什么Fifd               ifd jfidjf     fdifdF是呢  粉底 ！！~！@#@#~！@#￥%……&*（）——+「」「|：“《》？·-=【】、；‘，。、~!@#$%^&*()_+$￥|:"<>?`[]',./"""
  print("1")
  compute_f1(a, b)
  compute_em(a, b)
  compute_em(a, a)
  print("2")
  compute_f1_lcs(a, b)


if __name__ == '__main__':
  OPTS = parse_args()
  if OPTS.out_image_dir:
    import matplotlib

    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
  main()
  # test()
