from PIL import Image
import os
import time
from threading import Thread  # 创建线程的模块
import keras_segmentation
import sys
os.environ['CUDA_VISIBLE_DEVICES'] = ''

print(sys.argv)
input_img_path , output_mask_path, output_img_path = sys.argv[1], sys.argv[2], sys.argv[3]

start_time = time.time()
def segmentation(input_file_path, output_mask_path, output_file_path):
    model = keras_segmentation.pretrained.pspnet_101_voc12()  # load the pretrained model trained on Pascal VOC 2012 dataset
    out = model.predict_segmentation(
        inp=input_file_path,
        out_fname=output_mask_path
    )
    img1 = Image.open(input_file_path)
    img2 = Image.open(output_mask_path)
    img = Image.blend(img1.convert('RGB'), img2.convert('RGB'), alpha=0.5)
    img.save(output_file_path)
    print('finish segmentation')

out = segmentation(input_img_path, output_mask_path, output_img_path)
print("Segmentation Time cost:", time.time() - start_time)