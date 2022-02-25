import os
from xml.dom.minidom import Document
from functools import partial


def make_xml(img, prefix):
    id = os.path.splitext(img)[0]
    xml_name = os.path.join(prefix, id + '.xml')
    
    doc = Document()
    
    ann = doc.createElement('annotation')
    doc.appendChild(ann)
    
    folder = doc.createElement('folder')
    folder.appendChild(doc.createTextNode("JPEGImages"))
    
    filename = doc.createElement('filename')
    filename.appendChild(doc.createTextNode(img))
    
    ann.appendChild(folder)
    ann.appendChild(filename)
    
    with open(xml_name, 'w') as f:
        f.write(doc.toprettyxml())


if __name__ == '__main__':
    test_img_dir = 'D:/137/dataset/competitions/shipdet_dcic/xmy_data/test/VOC2012/JPEGImages'
    test_ann_dir = 'D:/137/dataset/competitions/shipdet_dcic/xmy_data/test/VOC2012/Annotations'
    test_txt = 'D:/137/dataset/competitions/shipdet_dcic/xmy_data/test.txt'

    if not os.path.exists(test_ann_dir):
        os.makedirs(test_ann_dir)

    imgs = sorted(os.listdir(test_img_dir))
    list(map(partial(make_xml, prefix=test_ann_dir), imgs))
    ids = list(map(lambda img: os.path.splitext(img)[0] + '\n', imgs))

    with open(test_txt, 'w') as f:
        f.writelines(ids)
