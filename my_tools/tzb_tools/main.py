# lib for my inference detector
import os
import sys
import argparse
import time
import json
import pdb

import torch
import numpy as np
import mmcv
from mmcv import Config, DictAction
from mmdet.apis import (async_inference_detector, inference_detector,
                        init_detector, show_result_pyplot)

import DOTA_devkit.polyiou as polyiou


def parse_args():
    cfg_name = 'cascade_mask_rcnn_swin_base_patch4_window7_mstrain_480-800_giou_4conv1f_adamw_3x_coco.py'
    ckpt_name = 'epoch_35.pth'
    thr = 0.988


    subsize = 768
    overlap = 200


    device = 'cuda:0'


    cfg_options = {'model.test_cfg.rcnn.score_thr': thr}
    cfg_file = os.path.join('/work/work_dirs', cfg_name)
    ckpt_file = os.path.join('/work/work_dirs', ckpt_name)


    parser = argparse.ArgumentParser()
    parser.add_argument('--subsize', default=subsize, help='size of splitted images')
    parser.add_argument('--overlap', default=overlap, help='overlap of splitted images')
    parser.add_argument("--input-dir", default='/input_path', help="input path", type=str)
    parser.add_argument("--output-dir", default='/output_path', help="output path", type=str)
    parser.add_argument('--config', default=cfg_file, help='Config file')
    parser.add_argument('--checkpoint', default=ckpt_file, help='Checkpoint file')
    parser.add_argument(
        '--device', default=device, help='Device used for inference')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        default=cfg_options,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')

    args = parser.parse_args()

    return args


def py_cpu_nms_poly_fast_np(dets, thresh):
    obbs = dets[:, 0:-1]
    x1 = np.min(obbs[:, 0::2], axis=1)
    y1 = np.min(obbs[:, 1::2], axis=1)
    x2 = np.max(obbs[:, 0::2], axis=1)
    y2 = np.max(obbs[:, 1::2], axis=1)
    scores = dets[:, 4]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)

    polys = []
    for i in range(len(dets)):
        tm_polygon = polyiou.VectorDouble([dets[i][0], dets[i][1],
                                           dets[i][2], dets[i][1],
                                           dets[i][2], dets[i][3],
                                           dets[i][0], dets[i][3]])
        polys.append(tm_polygon)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        ovr = []
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        hbb_inter = w * h
        hbb_ovr = hbb_inter / (areas[i] + areas[order[1:]] - hbb_inter)
        h_inds = np.where(hbb_ovr > 0)[0]
        tmp_order = order[h_inds + 1]
        for j in range(tmp_order.size):
            iou = polyiou.iou_poly(polys[i], polys[tmp_order[j]])
            hbb_ovr[h_inds[j]] = iou

        try:
            if np.isnan(ovr[0]):
                pdb.set_trace()
        except:
            pass
        inds = np.where(hbb_ovr <= thresh)[0]
        order = order[inds + 1]
    return keep


def inference_single(model,imagname, slide_size, chip_size,classnames):
    img = mmcv.imread(imagname)
    height, width, channel = img.shape
    slide_h, slide_w = slide_size
    hn, wn = chip_size

    # TODO: check the corner case
    total_detections = [np.zeros((0, 5)) for _ in range(len(classnames))]

    for i in range(int(width / slide_w + 1)):
        for j in range(int(height / slide_h) + 1):
            subimg = np.zeros((hn, wn, channel))
            chip = img[j * slide_h:j * slide_h + hn, i * slide_w:i * slide_w + wn, :3]
            subimg[:chip.shape[0], :chip.shape[1], :] = chip

            chip_detections = inference_detector(model, subimg)
            
            if isinstance(chip_detections[0], list):
                chip_detections = chip_detections[0]
            elif isinstance(chip_detections[0], np.ndarray):
                pass
            else:
                raise ValueError(f'type of chip_detections should be list or ndarray, not {type(chip_detections)}')

            for cls_id, name in enumerate(classnames):
                chip_detections[cls_id][:, 0:-1:2] += i * slide_w
                chip_detections[cls_id][:, 1:-1:2] += j * slide_h
                try:
                    total_detections[cls_id] = np.concatenate((total_detections[cls_id], chip_detections[cls_id]))
                except:
                    pdb.set_trace()

    for i in range(len(classnames)):
        keep = py_cpu_nms_poly_fast_np(total_detections[i], 0.5)
        total_detections[i] = total_detections[i][keep]
    return total_detections


if __name__ == '__main__':
    args = parse_args()

    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    print(f'score_thr={cfg.model.test_cfg.rcnn.score_thr}')
    print(f'subsize={args.subsize}, overlap={args.overlap}')


    print(f'{time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())}: initializing detector...')

    start_time = time.time()
    torch.backends.cudnn.benchmark = True

    classes = ['bigship']
    slide_size = (args.subsize - args.overlap, args.subsize - args.overlap)
    chip_size = (args.subsize, args.subsize)

    model = init_detector(cfg, args.checkpoint, device=args.device)
    input_img_list = os.listdir(os.path.join(args.input_dir,'img'))


    print(f'{time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())}: inferring results...')

    output_dicts = []

    for img_j, input_big_img in enumerate(input_img_list):
        print(f'{time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())}: {img_j + 1}/{len(input_img_list)}')

        input_big_img_path = os.path.join(args.input_dir, 'img', input_big_img)
        result = inference_single(model, input_big_img_path, slide_size, chip_size, classes)

        output_dict = {}
        output_dict.update({'image_name': input_big_img})
        labels = []

        for class_id, bbox_result in enumerate(result):
            if bbox_result.shape[0] != 0:
                for index in range(bbox_result.shape[0]):
                    x1 = bbox_result[index, 0]
                    y1 = bbox_result[index, 1]
                    x2 = bbox_result[index, 2]
                    y2 = bbox_result[index, 1]
                    x3 = bbox_result[index, 2]
                    y3 = bbox_result[index, 3]
                    x4 = bbox_result[index, 0]
                    y4 = bbox_result[index, 3]

                    points = [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
                    category_id = classes[class_id]
                    confidence = bbox_result[index, 4]
                    labels.append({'points': points, 'category_id': category_id, 'confidence': confidence})

        output_dict.update({'labels': labels})
        output_dicts.append(output_dict)

    print(f'{time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())}: writing results to json...')

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    output_filename = f'{args.output_dir}/ship_results.json'
    with open(output_filename, 'w') as json_file:
        json.dump(output_dicts, json_file, indent=4)

    print('done')
