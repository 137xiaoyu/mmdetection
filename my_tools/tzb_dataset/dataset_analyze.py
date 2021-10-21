import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from numpy.lib.function_base import average


def analyze_instance_attr():
    train_root = 'D:/137/dataset/tzb/input_path/train'
    img_path = os.path.join(train_root, 'img')
    gt_path = os.path.join(train_root, 'mask')
    img_list = sorted(glob.glob(os.path.join(img_path, '*.png')))
    gt_list = sorted(glob.glob(os.path.join(gt_path, '*.txt')))
    
    # read gt by instance
    instances = []
    instance_id = 0
    for index, gt in enumerate(gt_list):
        with open(gt, 'r') as file:
            data_list = file.read().split()
        for data in data_list:
            instance_id += 1
            cat = data.split(',')[0]
            bbox = data.split(',')[1:9]
            bbox = np.array(list(map(float, bbox)))
            instance = {'id': instance_id, 'image': img_list[index], 'category': cat, 'bbox': bbox}
            instances.append(instance)

    # analyze attr
    print(f'Total instances num: {len(instances)}')
    area_size_list = []
    for instance in instances:
        bbox = instance['bbox']
        xmin, ymin, xmax, ymax = min(bbox[0::2]), min(bbox[1::2]), max(bbox[0::2]), max(bbox[1::2])
        width, height = xmax - xmin, ymax - ymin
        area_size = width*height
        area_size_list.append(area_size)
        
    print(f'Max area size: {max(area_size_list):.2f}\n'
          f'Min area size: {min(area_size_list):.2f}\n'
          f'Average area size: {average(area_size_list):.2f}')
    plt.bar(range(len(instances)), area_size_list)
    plt.title('Area size of instances')
    plt.show()


if __name__ == '__main__':
    analyze_instance_attr()
