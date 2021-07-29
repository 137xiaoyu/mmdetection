import os
import json
import numpy as np
import sys
sys.path.append('/dev2/Fengjq/1grade/ship_det/mmdetection/fjq_workspace/tools/')

from utils_for_evaluate_eval_only import evaluate_two_jsons_with_different_confidence


if __name__ == '__main__':
    cfg_name = 'htc_swin_large_patch4_window7_mstrain_480-800_giou_4conv1f_adamw_3x_coco.py'
    ckpt_name = 'epoch_195.pth'


    file_root = os.path.abspath('work_dirs')
    cfg_file = os.path.join(file_root, cfg_name)
    ckpt_file = os.path.join(file_root, ckpt_name)
    
    input_dir = '/data/input_path'
    output_root = '/data/output_path_wcx'
    output_dir = os.path.join(output_root, os.path.splitext(os.path.basename(cfg_file))[0])

    label_json_file = '/dev2/Fengjq/1grade/ship_det/mmdetection/fjq_workspace/input_data/Data_root/val/bigship.json'
    predict_json_file = f'{output_dir}/ship_results.json'

    classes = ['bigship']

    confidence_all = []
    precision_all = []
    recall_all = []
    F1_all = []

    for i in np.linspace(0.5, 0.85, 8):
        print(f'{i:.2f}')
        precision, recall, F1 = evaluate_two_jsons_with_different_confidence(label_json_file, predict_json_file, i, 0.5)
        confidence_all.append(i)
        precision_all.append(precision)
        recall_all.append(recall)
        F1_all.append(F1)

    for i in np.linspace(0.86, 0.95, 10):
        print(f'{i:.2f}')
        precision, recall, F1 = evaluate_two_jsons_with_different_confidence(label_json_file, predict_json_file, i, 0.5)
        confidence_all.append(i)
        precision_all.append(precision)
        recall_all.append(recall)
        F1_all.append(F1)
    
    for i in np.linspace(0.955, 0.975, 5):
        print(f'{i:.3f}')
        precision, recall, F1 = evaluate_two_jsons_with_different_confidence(label_json_file, predict_json_file, i, 0.5)
        confidence_all.append(i)
        precision_all.append(precision)
        recall_all.append(recall)
        F1_all.append(F1)

    for i in np.linspace(0.976, 0.999, 24):
        print(f'{i:.3f}')
        precision, recall, F1 = evaluate_two_jsons_with_different_confidence(label_json_file, predict_json_file, i, 0.5)
        confidence_all.append(i)
        precision_all.append(precision)
        recall_all.append(recall)
        F1_all.append(F1)

    mF1 = np.mean(F1_all)
    print("\nconfidence   precision   recall   F1  ")
    for i, (c, p, r, f1) in enumerate(zip(confidence_all, precision_all, recall_all, F1_all)):
        print('{:.3f}        {:.3f}       {:.3f}    {:.3f}'.format(c, p[0], r[0], f1[0]))
    print("mF1 = {}".format(mF1))

    with open(predict_json_file, 'r') as file:
        preds = json.load(file)
    
    total_results_num = 0
    for pred in preds:
        total_results_num += len(pred['labels'])
    print(f'total_results_num = {total_results_num}\n')
