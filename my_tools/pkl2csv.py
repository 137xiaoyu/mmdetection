import argparse
import os
import pickle
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('test_txt', type=str)
parser.add_argument('pkl_file', type=str)
parser.add_argument('csv_file', type=str)

if __name__ == '__main__':
    args = parser.parse_args()

    with open(args.test_txt, 'r') as f:
        img_names = f.readlines()
    img_ids = list(map(lambda img_name: os.path.splitext(img_name.strip())[0], img_names))

    with open(args.pkl_file, 'rb') as f:
        results = pickle.load(f)

    with open(args.csv_file, 'w') as f:
        for id, bboxes in zip(img_ids, results):
            bboxes = np.vstack(bboxes)
            write_line = "{},".format(id)

            for bbox in bboxes:
                x0, y0 = (bbox[0] + bbox[2]) * 1.0 / 2, (bbox[1] + bbox[3]) * 1.0 / 2
                width = bbox[2] - bbox[0]
                height = bbox[3] - bbox[1]
                score = bbox[4]

                output = [x0 / 256, y0 / 256, width / 256, height / 256, score]
                write_line += "{:.9f} {:.9f} {:.9f} {:.9f} {:.9f};".format(*output)

            write_line = write_line.strip(";")
            write_line += "\n"
            f.write(write_line)
