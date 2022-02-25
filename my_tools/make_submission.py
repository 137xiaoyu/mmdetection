import argparse
import os
import pickle
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('test_txt', type=str)
parser.add_argument('results_file', type=str)
parser.add_argument('submission_file', type=str)
parser.add_argument('--score_thrs', type=float, nargs='+')


def apply_score_thr(result, thr):
    scores = result[:, -1]
    inds = scores > thr
    new_res = result[inds, :]
    return new_res


def apply_score_thrs(result, thrs):
    for thr in thrs:
        new_res = apply_score_thr(result, thr)
        if new_res.shape[0]:
            break

    return new_res


if __name__ == '__main__':
    args = parser.parse_args()

    with open(args.test_txt, 'r') as f:
        img_names = f.readlines()
    img_ids = list(map(lambda img_name: os.path.splitext(img_name.strip())[0], img_names))

    with open(args.results_file, 'rb') as f:
        results = pickle.load(f)

    submission = {}

    for res, id in zip(results, img_ids):
        submission[id] = []

        res = np.vstack(res)  # only one class
        new_res = apply_score_thrs(res, args.score_thrs)

        for bbox in new_res:
            x0, y0 = (bbox[0] + bbox[2]) * 1.0 / 2, (bbox[1] + bbox[3]) * 1.0 / 2
            width = bbox[2] - bbox[0]
            height = bbox[3] - bbox[1]
            submission[id].append([x0 / 256, y0 / 256, width / 256, height / 256])

    with open(args.submission_file, 'w') as f:
        for key, value in submission.items():
            write_line = "{},".format(key)
            for bbox in value:
                write_line += "{:.9f} {:.9f} {:.9f} {:.9f};".format(*bbox)
            write_line = write_line.strip(";")
            write_line += "\n"
            f.write(write_line)
