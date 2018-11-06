import sys
import os
import json
import glob
import xml.etree.ElementTree as ET


xml_path = r'D:\train-6000-check\xml-1000-aug'
fl=open(r"C:\Users\CuiWenbing\PycharmProjects\pre_pic\xml_list.txt","w")
for xml in glob.glob(xml_path+'/*.xml'):
    xml_name = os.path.basename(xml)
    fl.write(xml_name)
    fl.write('\n')
fl.close

def get_filename_as_int(filename):
    idx = len(img_name_to_id)
    idx += 1
    img_name_to_id[filename]=idx

# xmllist.txt为所有的xml的文件名，如1_a_A.xml, ...
test_list = r'C:\Users\CuiWenbing\PycharmProjects\pre_pic\xml_list.txt'

with open(test_list, 'r') as f_xml:
    xmls = f_xml.readlines()


xmls = [i.strip().split('.')[0] for i in xmls]

img_name_to_id={}
for i in xmls:
    get_filename_as_int(i)

def get(root, name):
    vars = root.findall(name)
    return vars

def get_and_check(root, name, length):
    vars = root.findall(name)
    if len(vars) == 0:
        raise NotImplementedError('Can not find %s in %s.'%(name, root.tag))
    if length > 0 and len(vars) != length:
        raise NotImplementedError('The size of %s is supposed to be %d, but is %d.'%(name, length, len(vars)))
    if length == 1:
        vars = vars[0]
    return vars

def get_id(filename):
    # try:
    #     data = img_name_to_id[filename.split('.')[0]]
    #     return data
    # except Exception as e:
    #     pass
    return img_name_to_id[filename.split('.')[0]]


def convert(xml_list, xml_dir, json_file):
    list_fp = open(xml_list, 'r')
    json_dict = {"images":[], "type": "instances", "annotations": [],
                 "categories": []}
    categories = PRE_DEFINE_CATEGORIES
    bnd_id = START_BOUNDING_BOX_ID
    for line in list_fp:
        line = line.strip()
        print("Processing %s"%(line))
        xml_f = os.path.join(xml_dir, line)
        tree = ET.parse(xml_f)
        root = tree.getroot()
        path = get(root, 'path')
        if len(path) == 1:
            filename = get(root,'filename')[0].text
            # print(name)
            # filename = os.path.basename(path[0].text)
        elif len(path) == 0:
            filename = get_and_check(root, 'filename', 1).text
        else:
            raise NotImplementedError('%d paths found in %s'%(len(path), line))
        ## The filename must be a number
        image_id = get_id(filename)
        size = get_and_check(root, 'size', 1)
        width = int(get_and_check(size, 'width', 1).text)
        height = int(get_and_check(size, 'height', 1).text)
        image = {'file_name': filename, 'height': height, 'width': width,
                 'id':image_id}
        json_dict['images'].append(image)
        ## Cruuently we do not support segmentation
        #  segmented = get_and_check(root, 'segmented', 1).text
        #  assert segmented == '0'
        for obj in get(root, 'object'):
            category = get_and_check(obj, 'name', 1).text
            if category not in categories:
                new_id = len(categories)
                categories[category] = new_id
            category_id = categories[category]
            bndbox = get_and_check(obj, 'bndbox', 1)
            xmin = int(get_and_check(bndbox, 'xmin', 1).text) - 1
            ymin = int(get_and_check(bndbox, 'ymin', 1).text) - 1
            xmax = int(get_and_check(bndbox, 'xmax', 1).text)
            ymax = int(get_and_check(bndbox, 'ymax', 1).text)
            assert(xmax > xmin)
            assert(ymax > ymin)
            o_width = abs(xmax - xmin)
            o_height = abs(ymax - ymin)
            ann = {'area': o_width*o_height, 'iscrowd': 0, 'image_id':
                   image_id, 'bbox':[xmin, ymin, o_width, o_height],
                   'category_id': category_id, 'id': bnd_id, 'ignore': 0,
                   'segmentation': []}
            json_dict['annotations'].append(ann)
            bnd_id = bnd_id + 1
    for cate, cid in categories.items():
        cat = {'supercategory': 'none', 'id': cid, 'name': cate}
        json_dict['categories'].append(cat)
    json_fp = open(json_file, 'w')
    json_str = json.dumps(json_dict)
    json_fp.write(json_str)
    json_fp.close()
    list_fp.close()

START_BOUNDING_BOX_ID = 1
PRE_DEFINE_CATEGORIES = {}
# 按需修改标签名称
# PRE_DEFINE_CATEGORIES = {"tuhen": 1, "qianao": 2, "zhezhou": 3, "tudian": 4,
#                           "qipao":5, "bianyuan": 6, "huahen": 7, "guahen": 8, "shenao": 9,
#                           "qingweiao": 10}
# PRE_DEFINE_CATEGORIES = {"yinlie": 1, "jiaochayinlie": 2,"shixiao":3}s
# PRE_DEFINE_CATEGORIES = {"yinlie": 1, "shixiao":2}
PRE_DEFINE_CATEGORIES = {"yinlie": 1, "shixiao":2}
# PRE_DEFINE_CATEGORIES = {"xuhan":1}

# 转json
this_ = 'train'  # 分别为train，val，test，trainval

xml_path = r'C:\Users\CuiWenbing\PycharmProjects\pre_pic\xml_list.txt'.format(this_)
ann_path = r'D:\train-6000-check\xml-1000-aug'  # 标签地址
out = r'C:\Users\CuiWenbing\PycharmProjects\pre_pic\voc_xxxx_{}.json'.format(this_)  # 默认格式为voc_xxxx_<this_>.json

# 执行voc2coco
convert(xml_path, ann_path, out)

# 根据json文件文件的id，重新写<this_>.txt
with open(xml_path, 'r') as f_x:
    s = f_x.readlines()

names = [i.strip().split('.')[0] for i in s]

train_dict = {}

for i in names:
    train_dict[i] = img_name_to_id[i]

# 对字典进行按值排序
f = zip(train_dict.keys(), train_dict.values())
ff = sorted(f)
out_list = []
for i in ff:
    out_list.append(i[0]+'\n')
# 保存为新的txt
with open(r'C:\Users\CuiWenbing\PycharmProjects\pre_pic\new.txt', 'w') as f_w:
    f_w.writelines(out_list)