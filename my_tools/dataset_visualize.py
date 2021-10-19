import os
import glob
import numpy as np
import json
import cv2


def generate_gt_json():
    train_root = 'D:/137/dataset/tzb/input_path/train'
    img_path = os.path.join(train_root, 'img')
    gt_path = os.path.join(train_root, 'mask')
    img_list = sorted(glob.glob(os.path.join(img_path, '*.png')))
    gt_list = sorted(glob.glob(os.path.join(gt_path, '*.txt')))

    # read gt by img
    gt_dict = {}
    for index, gt in enumerate(gt_list):
        with open(gt, 'r') as file:
            data_list = file.read().split()
        gt_dict[img_list[index]] = []
        instance_id = 0
        for data in data_list:
            instance_id += 1
            cat = data.split(',')[0]
            bbox = data.split(',')[1:9]
            instance = {'id_by_image': instance_id, 'category': cat, 'bbox': bbox}
            gt_dict[img_list[index]].append(instance)

    with open('gt_img.json', 'w') as file:
            json.dump(gt_dict, file, indent=4)
    
    # read gt by instance
    gt_dict = {}
    gt_dict['instances'] = []
    instance_id = 0
    for index, gt in enumerate(gt_list):
        with open(gt, 'r') as file:
            data_list = file.read().split()
        for data in data_list:
            instance_id += 1
            cat = data.split(',')[0]
            bbox = data.split(',')[1:9]
            instance = {'id': instance_id, 'image': img_list[index], 'category': cat, 'bbox': bbox}
            gt_dict['instances'].append(instance)

    with open('gt_instance.json', 'w') as file:
        json.dump(gt_dict, file, indent=4)


def visualize_gt():
    json_file = 'gt_img.json'
    bbox_img_path = 'bbox_img'
    instance_path = 'instance_img'
    if not os.path.exists(bbox_img_path):
        os.mkdir(bbox_img_path)
    if not os.path.exists(instance_path):
        os.mkdir(instance_path)
    
    with open(json_file, 'r') as file:
        gt = json.load(file)
    for img_name, instances in gt.items():
        print('Processing ', img_name)
        img = cv2.imread(img_name)
        img_vis = img.copy()
        for index, instance in enumerate(instances):
            cat = instance['category']
            bbox_list = list(map(float, instance['bbox']))
            bbox = np.array_split(np.asarray(bbox_list, dtype=float), 4, axis=0)
            
            instance_img = img[int(bbox[0][1]):int(bbox[2][1]), 
                               int(bbox[0][0]): int(bbox[2][0]), :].copy()
            cv2.imwrite(os.path.join(instance_path, img_name.split('\\')[-1].split('.')[0] + 
                                     '_' + str(index + 1).zfill(3) + '.png'), instance_img)
            
            img_vis = cv2.drawContours(img_vis, [np.int64(bbox)], -1, (255, 0, 255), thickness=5)
            img_vis = cv2.putText(img_vis, cat, (int(bbox[0][0]), int(bbox[0][1])),
                        cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 255, 0), thickness=4)
            
        cv2.imwrite(os.path.join(bbox_img_path, img_name.split('\\')[-1]), img_vis)


if __name__ == '__main__':
    generate_gt_json()
    # visualize_gt()
