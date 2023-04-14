# coding=utf-8
'''
@author: Xin Huang
@contact: xin.huang@nlpr.ia.ac.cn
@file:nlgeval_utils.py
@time:2021/1/12 
@desc:
'''
# import sys
# sys.path.append("/home/xhuang/work/nlg-eval")
# from nlgeval.pycocoevalcap.bleu.bleu import Bleu
# from nlgeval.pycocoevalcap.meteor.meteor import Meteor


# str/unicode stripping in Python 2 and 3 instead of `str.strip`.
def _strip(s):
    return s.strip()
def compute_metrics(hypo_file, ref_files, tgt_lang='en'):
    with open(hypo_file, 'r') as f:
        hyp_list = f.readlines()
    ref_list = []
    if not isinstance(ref_files, (tuple, list)):
        ref_files = [ref_files]
    for iidx, ref_file in enumerate(ref_files):
        with open(ref_file, 'r') as f:
            ref_list.append(f.readlines())
    ref_list = [list(map(_strip, refs)) for refs in zip(*ref_list)]
    refs = {idx: strippedlines for (idx, strippedlines) in enumerate(ref_list)}
    hyps = {idx: [lines.strip()] for (idx, lines) in enumerate(hyp_list)}
    assert len(refs) == len(hyps)
    scorers = [
        (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
        (Meteor(lang=tgt_lang), "METEOR")
    ]
    ret_scores = {}
    for scorer, method in scorers:
        score, scores = scorer.compute_score(refs, hyps)
        if isinstance(method, list):
            for sc, scs, m in zip(score, scores, method):
                # print("%s: %0.6f" % (m, sc))
                ret_scores[m] = sc
        else:
            # print("%s: %0.6f" % (method, score))
            ret_scores[method] = score
        if isinstance(scorer, Meteor):
            scorer.close()
    print("Bleu4:", ret_scores['Bleu_4'], "Meteor:", ret_scores['METEOR'])
    return ret_scores
