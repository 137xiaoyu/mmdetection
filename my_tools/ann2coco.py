import os
import cv2
import json
import shutil
import pandas as pd
import numpy as np
from PIL import Image
from PIL import ImageEnhance
from PIL import ImageChops


#数据转coco
class Fabric2COCO:

    def __init__(self,
                 coco_root,
                 is_mode="train"
                 ):
        self.coco_root = coco_root
        self.images = []
        self.annotations = []
        self.categories = []
        self.img_id = 0
        self.ann_id = 0
        self.is_mode = is_mode
        if not os.path.exists(os.path.join(self.coco_root, self.is_mode + '2017')):
            os.makedirs(os.path.join(self.coco_root, self.is_mode + '2017'))

    def to_coco(self, anno_file,img_dir):
        self._init_categories()
        anno_result= pd.read_json(open(anno_file,"r"))
        
        if self.is_mode == "train":
            anno_result = anno_result.head(int(anno_result['name'].count()*0.9))#取数据集前百分之90
        elif self.is_mode == "val":
            anno_result = anno_result.tail(int(anno_result['name'].count()*0.1)) 
        name_list=anno_result["name"].unique()#返回唯一图片名字
        for img_name in name_list:
            img_anno = anno_result[anno_result["name"] == img_name]#取出此图片的所有标注
            bboxs = img_anno["bbox"].tolist()#返回list
            defect_names = img_anno["category"].tolist()
            assert img_anno["name"].unique()[0] == img_name

            img_path=os.path.join(img_dir,img_name)
            img =cv2.imread(img_path)
            h,w,c=img.shape
            # #这种读取方法更快
            # img = Image.open(img_path)
            # w, h = img.size
            #print(w,h)
            #h,w=6000,8192
            self.images.append(self._image(img_path,h, w))

            self._cp_img(img_path)#复制文件路径
            if self.img_id % 200 == 0:
                print("处理到第{}张图片".format(self.img_id))
            for bbox, label in zip(bboxs, defect_names):
                annotation = self._annotation(label, bbox)
                self.annotations.append(annotation)
                self.ann_id += 1
            self.img_id += 1
        instance = {}
        instance['info'] = 'fabric defect'
        instance['license'] = ['none']
        instance['images'] = self.images
        instance['annotations'] = self.annotations
        instance['categories'] = self.categories
        return instance

    def _init_categories(self):
        #1，2，3，4个类别
        for v in range(1,5):
            print(v)
            category = {}
            category['id'] = v
            category['name'] = str(v)
            category['supercategory'] = 'defect_name'
            self.categories.append(category)

    def _image(self, path,h,w):
        image = {}
        image['height'] = h
        image['width'] = w
        image['id'] = self.img_id
        image['file_name'] = os.path.basename(path)#返回path最后的文件名
        return image

    def _annotation(self,label,bbox):
        area=(bbox[2]-bbox[0])*(bbox[3]-bbox[1])
        points=[[bbox[0],bbox[1]],[bbox[2],bbox[1]],[bbox[2],bbox[3]],[bbox[0],bbox[3]]]
        annotation = {}
        annotation['id'] = self.ann_id
        annotation['image_id'] = self.img_id
        annotation['category_id'] = label
        # annotation['segmentation'] = []# np.asarray(points).flatten().tolist()
        annotation['segmentation'] = [np.asarray(points).flatten().tolist()]
        annotation['bbox'] = self._get_box(points)
        annotation['iscrowd'] = 0
        annotation["ignore"] = 0
        annotation['area'] = area
        return annotation

    def _cp_img(self, img_path):
        shutil.copy(img_path, os.path.join(
            os.path.join(self.coco_root, self.is_mode + '2017'), os.path.basename(img_path)))
    def _get_box(self, points):
        min_x = min_y = np.inf
        max_x = max_y = 0
        for x, y in points:
            min_x = min(min_x, x)
            min_y = min(min_y, y)
            max_x = max(max_x, x)
            max_y = max(max_y, y)
        '''coco,[x,y,w,h]'''
        return [min_x, min_y, max_x - min_x, max_y - min_y]
    def save_coco_json(self, instance, save_path):
        import json
        with open(save_path, 'w') as fp:
            json.dump(instance, fp, indent=1, separators=(',', ': '))#缩进设置为1，元素之间用逗号隔开 ， key和内容之间 用冒号隔开


if __name__ == '__main__':
    dataroot = 'D:/137/dataset/gdgrid3'
    trainval_csv_name = os.path.join(dataroot, '3train_rname.csv')
    trainval_json_name = os.path.join(dataroot, 'training.json')
    test_csv_name = os.path.join(dataroot, '3_testa_user.csv')
    coco_dataroot = os.path.join(dataroot, 'coco')
    ann_path = os.path.join(coco_dataroot, 'annotations')
    
    if not os.path.exists(os.path.join(os.path.join(coco_dataroot, 'test2017'))):
        os.makedirs(os.path.join(os.path.join(coco_dataroot, 'test2017')))
    
    if not os.path.exists(ann_path):
        os.makedirs(ann_path)
    
    '''
    test
    '''
    df = pd.read_csv(test_csv_name)
    df_img_path = df['image_url']
    instance = {}
    instance['info'] = 'fabric defect'
    instance['license'] = ['none']
    instance['images'] = []
    instance['annotations'] = []
    instance['categories'] = []
    
    img_id = 0
    for img_name in df_img_path:
        if img_id % 200 == 0:
            print(f'{img_id}/{len(df_img_path)}')
        img_path = os.path.join(dataroot, img_name)
        img = cv2.imread(img_path)
        h, w, c = img.shape
        image = {}
        image['height'] = h
        image['width'] = w
        image['id'] = img_id
        image['file_name'] = os.path.basename(img_path) # 返回path最后的文件名
        instance['images'].append(image)
        shutil.copy(img_path, os.path.join(
            os.path.join(os.path.join(coco_dataroot, 'test2017')), os.path.basename(img_path)))
        img_id += 1
    
    with open(os.path.join(ann_path, 'instances_test2017.json'), 'w') as test_json_file:
        json.dump(instance, test_json_file, indent=1, separators=(',', ': '))
    
    '''
    trainval
    '''
    df = pd.read_csv(trainval_csv_name, header=None)
    df_img_path = df[4]
    df_img_mark = df[5]
    #统计一下类别,并且重新生成原数据集标注文件，保存到json文件中
    dict_class = {
        "badge":0,
        "offground":0,
        "ground":0,
        "safebelt":0
    }
    dict_lable = {
        "badge":1,
        "offground":2,
        "ground":3,
        "safebelt":4
    }
    data_dict_json = []
    image_width,image_height = 0,0
    ids = 0
    false = False #将其中false字段转化为布尔值False
    true = True #将其中true字段转化为布尔值True
    for img_id, one_img in enumerate(df_img_mark):
        if img_id % 200 == 0:
            print(f'{img_id}/{len(df_img_mark)}')
        one_img = eval(one_img)["items"]
        # print(one_img)
        # print(one_img["items"])
        one_img_name = df_img_path[img_id]
        img = Image.open(os.path.join(dataroot,one_img_name))
        ids = ids + 1
        w, h = img.size
        image_width += w
        image_height += h
        # print(one_img_name)
        
        for one_mark in one_img:
            # print(one_mark)
            one_label = one_mark["labels"]['标签']
            #print(one_mark)
            try:
                dict_class[str(one_label)] += 1
                # category = str(one_label)
                category = dict_lable[str(one_label)]
                bbox = one_mark["meta"]["geometry"]
            except:
                dict_class["badge"] += 1#标签为"监护袖章(红only)"表示类别"badge"
                # category = "badge"
                category = 1
                bbox = one_mark["meta"]["geometry"]
        
            one_dict = {}
            one_dict["name"] = str(one_img_name)
            one_dict["category"] = category
            one_dict["bbox"] = bbox
            data_dict_json.append(one_dict)
    print(image_height/ids,image_width/ids)
    print(dict_class)
    print(len(data_dict_json))
    print(data_dict_json[0])
    with open(trainval_json_name, 'w') as fp:
        json.dump(data_dict_json, fp, indent=1, separators=(',', ': '))#缩进设置为1，元素之间用逗号隔开 ， key和内容之间 用冒号隔开
        fp.close()
    
    '''转换为coco格式'''
    #训练集,划分90%做为训练集
    fabric2coco = Fabric2COCO(coco_dataroot)
    train_instance = fabric2coco.to_coco(trainval_json_name, dataroot)
    fabric2coco.save_coco_json(train_instance, 
                               os.path.join(ann_path, 'instances_{}2017.json'.format("train")))

    
    '''转换为coco格式'''
    #验证集，划分10%做为验证集
    fabric2coco = Fabric2COCO(coco_dataroot, is_mode = "val")
    val_instance = fabric2coco.to_coco(trainval_json_name, dataroot)
    fabric2coco.save_coco_json(val_instance, 
                               os.path.join(ann_path, 'instances_{}2017.json'.format("val")))
