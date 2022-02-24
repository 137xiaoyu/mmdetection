from os.path import basename, splitext


def get_img_id(line):
    line = line.strip()
    img = line.split(' ')[0]
    img_id = splitext(basename(img))[0]
    return img_id + '\n'


if __name__ == '__main__':
    train_txt = 'D:/137/dataset/competitions/shipdet_dcic/xmy_data/train_src.txt'
    val_txt = 'D:/137/dataset/competitions/shipdet_dcic/xmy_data/val_src.txt'

    new_train_txt = 'D:/137/dataset/competitions/shipdet_dcic/xmy_data/train.txt'
    new_val_txt = 'D:/137/dataset/competitions/shipdet_dcic/xmy_data/val.txt'

    with open(train_txt, 'r') as f:
        train_ann = f.readlines()
    with open(val_txt, 'r') as f:
        val_ann = f.readlines()

    new_train_ann = list(map(get_img_id, train_ann))
    new_val_ann = list(map(get_img_id, val_ann))

    with open(new_train_txt, 'w') as f:
        f.writelines(new_train_ann)
    with open(new_val_txt, 'w') as f:
        f.writelines(new_val_ann)
