import argparse
import cv2
import numpy as np
import os

parser = argparse.ArgumentParser()
parser.add_argument('csv_file', type=str)
parser.add_argument('img_path', type=str)
parser.add_argument('save_path', type=str)
parser.add_argument('--mode', type=str, choices=['all', 'empty'], default='all')


def xywh2pts(bbox):
    x0, y0, width, height = bbox[:]
    pts = np.array([x0 - width / 2, y0 - height / 2, x0 + width / 2, y0 + height / 2]) * 256
    return pts


def parse_line(line):
    line = line.strip()
    img_id, dets = line.split(',')

    img_name = img_id + '.jpg'
    bboxes = []

    if dets:
        dets = dets.split(';')
        for det in dets:
            det = det.split(' ')
            det = list(map(float, det))
            bbox = xywh2pts(det)

            bboxes.append(bbox)

    return img_name, bboxes


if __name__ == '__main__':
    args = parser.parse_args()

    with open(args.csv_file, 'r') as f:
        results = f.readlines()

    results = list(map(parse_line, results))

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    for result in results:
        img_name, bboxes = result

        if args.mode == 'all':
            img = cv2.imread(os.path.join(args.img_path, img_name))
            for bbox in bboxes:
                bbox = bbox.astype(int)
                cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255),
                              thickness=3)  # B G R

            cv2.imwrite(os.path.join(args.save_path, img_name), img)

        elif args.mode == 'empty':
            if not bboxes:
                img = cv2.imread(os.path.join(args.img_path, img_name))
                cv2.imwrite(os.path.join(args.save_path, img_name), img)
