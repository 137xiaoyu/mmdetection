import os
import glob
import numpy as np
import json
import cv2


def gaussian_blur(img):
    """Gaussian blur preprocessing

    Args:
        img (ndarray): input img

    Returns:
        ndarray: blurred img
    """
    return cv2.GaussianBlur(img, (9, 9), 10)


def dataset_preprocess(data_root, save_path, preprocessing_method, sub_size=768):
    """Split, preprocess and save the subimgs containing target object

    Args:
        data_root (str): data root of the original dataset
        save_path (str): save path for subimgs and object instances
        preprocessing_method (func): preprocessing method function
        sub_size (int, optional): sub size of subimgs. Defaults to 768.
    """
    img_path = os.path.join(data_root, 'img')
    gt_path = os.path.join(data_root, 'mask')
    img_list = sorted(glob.glob(os.path.join(img_path, '*.png')))
    gt_list = sorted(glob.glob(os.path.join(gt_path, '*.txt')))

    save_path_subimg = os.path.join(save_path, 'subimg')
    save_path_instance = os.path.join(save_path, 'instance')
    save_path_subimg_processed = os.path.join(save_path, 'subimg_processed')
    save_path_instance_processed = os.path.join(save_path, 'instance_processed')
    
    # make dirs
    save_paths = [save_path_subimg, save_path_instance, save_path_subimg_processed, save_path_instance_processed]
    for one_save_path in save_paths:
        if not os.path.exists(one_save_path):
            os.makedirs(one_save_path)

    # read ground truth
    gt_dict = {}
    for index_img, gt in enumerate(gt_list):
        with open(gt, 'r') as file:
            data_list = file.read().split()
            
        # read gt of one img
        one_img_gt_dict = {}
        for data in data_list:
            # category
            cat = data.split(',')[0]
            
            # bbox = [x1, y1, x2, y2, x3, y3, x4, y4]
            # (x1, y1) is the left-top point
            # (x2, y2) is the right-top point
            # (x3, y3) is the right-bottom point
            # (x4, y4) is the left-bottom point
            bbox = data.split(',')[1:9]
            bbox = np.array(list(map(float, bbox)))
            
            # (i, j) means the (row, column) respectively
            i = int((bbox[1] + bbox[5]) / 2 // sub_size)
            j = int((bbox[0] + bbox[4]) / 2 // sub_size)
            
            bbox[1::2] -= i*sub_size
            bbox[0::2] -= j*sub_size
            bbox[0] = max(0, bbox[0])
            bbox[6] = max(0, bbox[6])
            bbox[1] = max(0, bbox[1])
            bbox[3] = max(0, bbox[3])
            bbox = list(bbox)
            
            instance = {'category': cat, 'bbox': bbox}
            
            # store gt according to subimg
            if (i, j) in one_img_gt_dict:
                one_img_gt_dict[(i, j)].append(instance)
            else:
                one_img_gt_dict[(i, j)] = [instance]
            
        gt_dict[img_list[index_img]] = one_img_gt_dict

    # preprocess imgs and save
    for index_img, (img_name, one_img_gt_dict) in enumerate(gt_dict.items()):
        print(f'Processing img: {index_img + 1}/{len(gt_dict)}')
        img = cv2.imread(img_name)
        
        # process one img
        for (i, j), instances in one_img_gt_dict.items():
            subimg_prefix = f'{index_img + 1}_{i*768}_{j*768}'
            subimg = img[i*sub_size:(i + 1)*sub_size, j*sub_size:(j + 1)*sub_size, :].copy()
            subimg_processed = preprocessing_method(subimg)
            
            # process one subimg
            for index_instance, instance in enumerate(instances):
                # bbox = np.array([[x1, y1], [x2, y2], [x3, y3], [x4, y4]])
                bbox = np.array_split(np.asarray(instance['bbox']), 4)
                
                instance_img = subimg[int(bbox[0][1]):int(bbox[2][1]), int(bbox[0][0]):int(bbox[2][0]), :].copy()
                instance_img_processed = subimg_processed[int(bbox[0][1]):int(bbox[2][1]), int(bbox[0][0]):int(bbox[2][0]), :].copy()
                
                instance_img_name = os.path.join(save_path_instance, subimg_prefix + f'_{index_instance + 1}.png')
                instance_img_processed_name = os.path.join(save_path_instance_processed, subimg_prefix + f'_{index_instance + 1}.png')
                cv2.imwrite(instance_img_name, instance_img)
                cv2.imwrite(instance_img_processed_name, instance_img_processed)
                
                subimg = cv2.drawContours(subimg, [np.int64(bbox)], -1, (255, 0, 255), thickness=5)
                subimg_processed = cv2.drawContours(subimg_processed, [np.int64(bbox)], -1, (255, 0, 255), thickness=5)
                
            subimg_name = os.path.join(save_path_subimg, subimg_prefix + '.png')
            subimg_processed_name = os.path.join(save_path_subimg_processed, subimg_prefix + '.png')
            cv2.imwrite(subimg_name, subimg)
            cv2.imwrite(subimg_processed_name, subimg_processed)


if __name__ == '__main__':
    data_root = 'D:/137/dataset/tzb/input_path/train/'
    save_path = 'D:/137/dataset/tzb/preprocessing_result/'
    sub_size = 768
    dataset_preprocess(data_root, save_path, gaussian_blur, sub_size=sub_size)
