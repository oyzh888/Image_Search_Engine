from threading import Thread  # 创建线程的模块
from imageai.Detection import ObjectDetection
import tensorflow as tf
import keras.backend as K
import sys
import time

print(sys.argv)
input_img_path , output_img_path = sys.argv[1], sys.argv[2]
def image_detection(input_img_path, output_img_path):

    start_time = time.time()
    # def task(name=None):
    # with tf.Session(graph=tf.Graph()) as sess:
    #     K.set_session(sess)
    # with tf.get_default_graph().as_default():
    detector = ObjectDetection()
    detector.setModelTypeAsRetinaNet()
    detector.setModelPath('/data/ouyangzhihao/Web/Model/Detection/resnet50_coco_best_v2.0.1.h5')
    detector.loadModel()

    detections = detector.detectObjectsFromImage(input_image=input_img_path,output_image_path=output_img_path)
    print('thread time:', time.time() - start_time )
    # p = Thread(target=task, args=('线程1',))
    # p.start()  # 只是给操作系统发送了一个就绪信号，并不是执行。操作系统接收信号后安排cpu运行
    print('Finish image_detection')

image_detection(input_img_path, output_img_path)