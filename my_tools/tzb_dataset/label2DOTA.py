import os
import glob


def label_convert():
    dataroot = 'D:/137/dataset/tzb'
    train_path = os.path.join(dataroot, 'input_path/train')
    img_path = os.path.join(train_path, 'img')
    mask_path = os.path.join(train_path, 'mask')
    save_label_path = os.path.join(train_path, 'labelTxt')
    
    if not os.path.exists(save_label_path):
        os.mkdir(save_label_path)
        
    label_list = glob.glob(os.path.join(mask_path, '*.txt'))
    
    for label_file in label_list:
        print(label_file)
        with open(label_file, 'r') as file_to_read:
            labels = file_to_read.read().split('\n')
            with open(os.path.join(save_label_path, os.path.basename(label_file)), 'w') as file:
                file.write("'imagesource': tzb\n")
                file.write("'gsd': 5\n")
                for label in labels:
                    if not label:
                        continue
                    cat, x1, y1, x2, y2, x3, y3, x4, y4 = label.split(',')
                    sep = ' '   # 分隔符
                    seq = [x1, y1, x2, y2, x3, y3, x4, y4, cat, '0']    # difficult设置为0
                    line_to_write = sep.join(seq) + '\n'
                    file.write(line_to_write)


if __name__ == '__main__':
    label_convert()
