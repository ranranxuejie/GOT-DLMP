import json
# from doctextVQAeval import VQAEval
import nltk
nltk.download('wordnet')
import argparse
# import fitz as pymupdf
import nltk
from nltk.metrics import precision, recall, f_measure
import numpy as np
import jieba
# import megfile as mf
import pickle
import pandas as pd
import re
# from loguru import logger
# nltk.download('wordnet')
from nltk.translate import meteor_score

# from marker_scoring import score_text
# from utils import contain_chinese_string

gt_path = './eval.json'
# out_path = './result_checkpoint-8000-encoder.json'
out_path = './result_GOT_weights.json'
save_csv_path = './org_result'

def preprocess(text, predict_root_):
    if 'InternVL' in predict_root_:
        text = text.split("All words in the image:\n")[1]
        text = text.split("[UNUSED_TOKEN_145]")[0]
    return text


def contain_chinese_string(text):
    # 使用正则表达式匹配中文字符
    chinese_pattern = re.compile(r'[\u4e00-\u9fa5]')
    return bool(chinese_pattern.search(text))


inline_reg = re.compile(r"\\\((.*?)(?<!\\)\\\)")
display_reg = re.compile(r"\\\[(.+?)(?<!\\)\\\]")
table_reg = re.compile(r"\\begin\{tabular\}(.+?)(?:\\end\{tabular\}|$)", re.S)


def split_text( a_type):
    """
    Split a list of pages into text, inline math, display math, and table blocks.
    Args:
        pages: The pages to split.
    """
    if a_type == 'label':
        pages = json.load(open(gt_path, encoding='utf-8'))
    else:
        pages = json.load(open(out_path, encoding='utf-8'))
    img_name,text, math, table = [], [], [], []
    for page in pages:
        for i, reg in enumerate([inline_reg, display_reg, table_reg]):
            if a_type == 'label':
                label_str = page["conversations"][1]['value']
            else:
                label_str = pages[page]
            matches = "\n".join(reg.findall(label_str))
            if i == 2:
                table.append(matches)
            elif i == 1:
                math[-1] += matches
            else:
                math.append(matches)
        # page_str = label_str
        text.append(label_str.strip())
        img_name.append(page)
    return text, math, table, img_name


def nougat_per_metrics(predict_root_, pred, gt, minlen=1, heavy_mode: int = 2):
    """
    Args:
    - heavy_mode:
        0 is clean mode, only similar, bleu, f1
        1 is normal, do not include edit_dist
        2 is heavy, total
    """
    metrics = {}

    # pred = preprocess(pred, predict_root_)

    if len(pred) < minlen or len(gt) < minlen:
        return metrics

    # metrics["similar"] = score_text(pred, gt)
    if contain_chinese_string(gt) or contain_chinese_string(pred):
        reference = jieba.lcut(gt)
        hypothesis = jieba.lcut(pred)
    else:
        reference = gt.split()
        hypothesis = pred.split()

    metrics["bleu"] = nltk.translate.bleu([reference], hypothesis)
    if heavy_mode >= 1:
        # try:
        metrics["meteor"] = meteor_score.meteor_score([reference], hypothesis)
        # except LookupError:
        #     metrics["meteor"] = np.nan

    reference = set(reference)
    hypothesis = set(hypothesis)
    metrics["f_measure"] = f_measure(reference, hypothesis)

    if heavy_mode >= 1:
        metrics["precision"] = precision(reference, hypothesis)
        metrics["recall"] = recall(reference, hypothesis)
    if heavy_mode == 2:
        # 速度太慢
        metrics["edit_dist"] = nltk.edit_distance(pred, gt) / max(len(pred), len(gt))
    return metrics


def doc_formated_text_eval():
    # predicts = json.load(open(predict_root_, encoding='utf-8'))

    # print(predicts)

    gt_text_split, gt_math_split, gt_table_split, img_name = split_text( 'label')
    pre_text_split, pre_math_split, pre_table_split,_ = split_text( 'answer')
    text_results = []
    math_results = []
    table_results = []
    data_i = 0
    for gt0, pre0, gt1, pre1, gt2, pre2 in zip(gt_text_split, pre_text_split, gt_math_split, pre_math_split,
                                               gt_table_split, pre_table_split):
        # try:
        # text, math, table
        print(data_i)
        data_i += 1
        text_gts, text_pres = gt0, pre0
        math_gts, math_pres = gt1, pre1
        table_gts, table_pres = gt2, pre2

        # for text_gt, text_pre in zip(text_gts, text_pres):
        ans = nougat_per_metrics(out_path, text_gts, text_pres)
        # if len(ans) == 0:
        #     continue
        if ans:
            text_results.append(ans)
        # for math_gt, math_pre in zip(math_gts, math_pres):
        ans = nougat_per_metrics(out_path, math_gts, math_pres)
        # if len(ans) == 0:
        #     continue
        if ans:
            math_results.append(ans)

        # for table_gt, table_pre in zip(table_gts, table_pres):
        ans = nougat_per_metrics(out_path, table_gts, table_pres)
        # if len(ans) == 0:
        #     continue
        if ans:
            table_results.append(ans)

    mean_dict = {}
    # print((result))
    # print(len(result))
    mean_dict["eval question num"] = len(text_results)
    mean_dict['text'] = {}
    mean_dict['math'] = {}
    mean_dict['table'] = {}

    for k, v in text_results[0].items():
        mean_dict['text'][k] = 0
        mean_dict['math'][k] = 0
        mean_dict['table'][k] = 0

    for each in text_results:
        for k, v in each.items():
            mean_dict['text'][k] += v

    for each in math_results:
        for k, v in each.items():
            mean_dict['math'][k] += v

    for each in table_results:
        for k, v in each.items():
            mean_dict['table'][k] += v

    for k, v in mean_dict['text'].items():
        if len(text_results) != 0:
            mean_dict['text'][k] /= len(text_results)

    for k, v in mean_dict['math'].items():
        if len(math_results)!= 0:
            mean_dict['math'][k] /= len(math_results)

    for k, v in mean_dict['table'].items():
        if len(table_results)!= 0:
            mean_dict['table'][k] /= len(table_results)

    print(json.dumps(mean_dict, indent=4))
    math_results = pd.DataFrame(math_results)
    # math_results['img_name'] = img_name
    table_results = pd.DataFrame(table_results)
    # table_results['img_name'] = img_name
    text_results = pd.DataFrame(text_results)
    text_results['img_name'] = img_name
    result_path = save_csv_path
    math_results.to_csv(result_path+'/math_results.csv')
    table_results.to_csv(result_path+'/table_results.csv')
    text_results.to_csv(result_path+'/text_results.csv')


def doc_text_eval(gt_root_, predict_root_, datatype):
    predicts = json.load(open(predict_root_, encoding='utf-8'))

    # print(predicts)
    result = []
    for ann in predicts:
        try:
            ans = nougat_per_metrics(predict_root_, ann["label"], ann["answer"])
            if len(ans) == 0:
                continue
            result.append(ans)
        except:
            assert False, print("ERROR!!! Check yout output!!!")

    mean_dict = {}
    # print((result))
    # print(len(result))
    mean_dict["eval question num"] = len(result)
    for k, v in result[0].items():
        mean_dict[k] = 0

    for each in result:
        for k, v in each.items():
            mean_dict[k] += v

    for k, v in mean_dict.items():
        if k == "eval question num":
            continue
        mean_dict[k] /= len(result)
    print(json.dumps(mean_dict, indent=4))


# doc_text_eval("/data/data/DocVQA/val/val_v1.0.json", "/data/codes/GOT_docshot-main/results_cc595k-freeze-docvqa-unfreeze-224/results_final.json", "Doc")


doc_formated_text_eval()

# doc_text_eval(args.gt_path, args.out_path + "/results_final.json", args.datatype)
