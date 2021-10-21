import argparse
import os
import my_tools.ImgSplit_multi_process as ImgSplit_multi_process
import DOTA_devkit.SplitOnlyImage_multi_process as SplitOnlyImage_multi_process
from my_tools.DOTA2COCO import DOTA2COCO
from multiprocessing import Pool
import shutil


wordname_1 = ['bigship']


def parse_args():
    data_root = 'D:/137/dataset/tzb'
    src_path = os.path.join(data_root, 'input_path')
    dst_path = os.path.join(data_root, 'input_path_coco')
    img_size = 768
    overlap = 200
    num_workers = 32
    
    parser = argparse.ArgumentParser(description='make dataset DOTA2COCO')
    parser.add_argument('--srcpath', default=src_path)
    parser.add_argument('--dstpath', default=dst_path)
    parser.add_argument('--img-size', default=img_size)
    parser.add_argument('--overlap', default=overlap)
    parser.add_argument('--num-workers', default=num_workers)
    args = parser.parse_args()

    return args


def prepare(srcpath, dstpath, img_size, overlap, num_workers):
    """
    :param srcpath: train, val, test
          train --> train2017, val --> val2017, test --> test2017
    :return:
    """
    
    src_train_path = os.path.join(srcpath, 'train')
    src_val_path = os.path.join(srcpath, 'val')
    src_test_path = os.path.join(srcpath, 'test')
    
    split_train_path = os.path.join(dstpath, f'split/train_{img_size}_{overlap}')
    split_val_path = os.path.join(dstpath, f'split/val_{img_size}_{overlap}')
    split_test_path = os.path.join(dstpath, f'split/test_{img_size}_{overlap}')
    
    dst_train_img_path = os.path.join(dstpath, 'train2017')
    dst_val_img_path = os.path.join(dstpath, 'val2017')
    dst_test_img_path = os.path.join(dstpath, 'test2017')
    
    dst_train_ann_path = os.path.join(dstpath, 'annotations')
    dst_val_ann_path = os.path.join(dstpath, 'annotations')
    dst_test_ann_path = os.path.join(dstpath, 'annotations')
    
    if not os.path.exists(split_train_path):
        os.makedirs(split_train_path)
    if not os.path.exists(split_val_path):
        os.makedirs(split_val_path)
    if not os.path.exists(split_test_path):
        os.makedirs(split_test_path)
    
    if not os.path.exists(dst_train_ann_path):
        os.makedirs(dst_train_ann_path)
    if not os.path.exists(dst_val_ann_path):
        os.makedirs(dst_val_ann_path)
    if not os.path.exists(dst_test_ann_path):
        os.makedirs(dst_test_ann_path)

    print('splitting training imgs...')
    split_train = ImgSplit_multi_process.splitbase(src_train_path,
                                                   split_train_path,
                                                   gap=overlap,
                                                   subsize=img_size,
                                                   num_process=num_workers)
    split_train.splitdata(1)
    print('splitting validation imgs...')
    split_val = ImgSplit_multi_process.splitbase(src_val_path,
                                                 split_val_path,
                                                 gap=overlap,
                                                 subsize=img_size,
                                                 num_process=num_workers)
    split_val.splitdata(1)
    print('splitting testing imgs...')
    split_test = SplitOnlyImage_multi_process.splitbase(os.path.join(src_test_path, 'img'),
                                                        os.path.join(split_test_path, 'img'),
                                                        gap=overlap,
                                                        subsize=img_size,
                                                        num_process=num_workers)
    split_test.splitdata(1)
    
    print('generating traning annotations...')
    DOTA2COCO(split_train_path, os.path.join(dst_train_ann_path, 'instances_train2017.json'))
    print('generating validation annotations...')
    DOTA2COCO(split_val_path, os.path.join(dst_val_ann_path, 'instances_val2017.json'))
    print('generating testing annotations...')
    DOTA2COCO(split_test_path, os.path.join(dst_test_ann_path, 'instances_test2017.json'), is_train=False)
    
    print('moving imgs...')
    if os.path.exists(dst_train_img_path):
        shutil.rmtree(dst_train_img_path)
    if os.path.exists(dst_val_img_path):
        shutil.rmtree(dst_val_img_path)
    if os.path.exists(dst_test_img_path):
        shutil.rmtree(dst_test_img_path)
    os.rename(os.path.join(split_train_path, 'img'), dst_train_img_path)
    os.rename(os.path.join(split_val_path, 'img'), dst_val_img_path)
    os.rename(os.path.join(split_test_path, 'img'), dst_test_img_path)


if __name__ == '__main__':
    args = parse_args()
    
    print(f'src path: {args.srcpath}\n' +
          f'dst path: {args.dstpath}\n' +
          f'img size: {args.img_size}\n' +
          f'overlap: {args.overlap}\n' +
          f'num_workers: {args.num_workers}')
    
    prepare(args.srcpath, args.dstpath, args.img_size, args.overlap, args.num_workers)
