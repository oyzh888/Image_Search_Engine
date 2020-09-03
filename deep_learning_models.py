from pycocotools.coco import COCO
import pickle
from feature_extractor import FeatureExtractor
import json
import numpy as np
import time
import os
from threading import Thread

dataDir='/data/ouyangzhihao/dataset/MSCOCO'
dataType='train2014'
annFile='{}/annotations/instances_{}.json'.format(dataDir,dataType)

# initialize COCO api for caption annotations
annFile = '{}/annotations/captions_{}.json'.format(dataDir,dataType)
coco = None
coco_caps = None


def if_one_variable_is_none(*args):
    for ele in args:
        if ele is None: return True
    return False

def init_COCO():
    # initialize COCO api for instance annotations
    start_time = time.time()
    global coco, coco_caps
    coco = COCO(annFile)
    coco_caps = COCO(annFile)
    print('Load COCO time:', time.time() - start_time)

def get_file_caption(file_name):
    if if_one_variable_is_none(coco, coco_caps): init_COCO()

    file_id = int(file_name[-4 - 12:-4])
    img = coco.loadImgs(file_id)[0]
    annIds = coco_caps.getAnnIds(imgIds=img['id']);
    anns = coco_caps.loadAnns(annIds)
    captions = coco_caps.showAnns(anns)
    return captions


# Classification
# Read image features
feature_path = '/data/ouyangzhihao/Web/Model/Results/COCO_Norm_Feature_MobileNet/features.pkl'
image_path = '/data/ouyangzhihao/Web/Model/Results/COCO_Norm_Feature_MobileNet/file_names.pkl'
# Files are in static folder
dataset_path = 'dataset/MSCOCO/'


# Read Imagenet classes
fe = features= img_paths = class_idx = None
def init_image_model():
    start_time = time.time()
    global fe, features, img_paths, class_idx
    fe = FeatureExtractor()
    features = pickle.load(open(feature_path, 'rb'))
    img_paths = pickle.load(open(image_path, 'rb'))
    img_paths = [dataset_path + var for var in img_paths]
    with open("imagenet_class_index.json", 'r') as f:
        class_idx = json.load(f)
    print('Load feature time:', time.time() - start_time)


def get_img_html_info(path, dist, probability, categories, captions):
    return {
        'path':path,
        'dist':dist,
        'probability':probability,
        'categories':categories,
        'captions':captions,
    }

def return_model_result(img, top_k):
    start_time = time.time()
    if if_one_variable_is_none(fe, features, img_paths, class_idx): init_image_model()
    query, prediction = fe.extract(img)
    query = np.reshape(query, -1)
    dists = np.linalg.norm(features - query, axis=-1)  # Do search
    ids = np.argsort(dists)[:top_k]  # Top k results

    # ImageNet
    # import ipdb; ipdb.set_trace()
    # print(prediction)

    pre_ids = prediction.argsort()[-3:]
    # descending -> ascending
    pre_ids = pre_ids[::-1]
    topk_class = [class_idx[str(id)][1] for id in pre_ids]
    res_to_html = []
    for id in ids:
        res_to_html += [get_img_html_info(
            path=img_paths[id], dist=dists[id], probability=prediction[pre_ids[0]],
            categories=topk_class, captions=get_file_caption(img_paths[id])
        )]
    print('classification time: ',time.time() - start_time)
    return res_to_html

# Image Detection
def image_detection(input_img_path, output_img_path):
    def task(name=None):
        os.system("python image_detection.py %s %s" % (input_img_path, output_img_path))
    p = Thread(target=task, args=('线程1',))
    p.start()  # 只是给操作系统发送了一个就绪信号，并不是执行。操作系统接收信号后安排cpu运行

# Image Segmentation
def image_segmentation(input_img_path, output_mask_path, output_img_path):
    def task(name=None):
        os.system("python image_segmentation.py %s %s %s" % (input_img_path, output_mask_path, output_img_path))
    p = Thread(target=task, args=('线程1',))
    p.start()  # 只是给操作系统发送了一个就绪信号，并不是执行。操作系统接收信号后安排cpu运行
