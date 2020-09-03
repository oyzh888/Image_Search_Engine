import os
import numpy as np
from PIL import Image

import glob
from datetime import datetime
from flask import Flask, request, render_template, redirect,url_for
from deep_learning_models import *
from gevent.pywsgi import WSGIServer

os.environ['CUDA_VISIBLE_DEVICES'] = ''
app = Flask(__name__, static_url_path='', root_path='')
server_root_path = os.getcwd()

detection_output_img_path_html = ""
segmentaion_output_img_path_html = ""
segmentaion_output_mask_path_html = ""

res_to_html = []
query_path = ""
img = None
new_file_name = upload_true_path = ""

def save_request_file():
    try:
        global img, new_file_name
        file = request.files['query_img']
        img = Image.open(file.stream)  # PIL image

        new_file_name = datetime.now().isoformat() + "_" + file.filename
        global query_path, upload_true_path
        query_path = "uploaded/" + new_file_name
        upload_true_path = os.path.join(server_root_path, "static", query_path)
        img.save(upload_true_path)
    except:
        return redirect(url_for('error'))

def send_img_post_to_result_page():
    try:
        global res_to_html
        res_to_html = return_model_result(img, top_k=9)

        # Start image detection

        detection_output_img_path = os.path.join(server_root_path, "static", 'detection_result', new_file_name)
        image_detection(upload_true_path, output_img_path=detection_output_img_path)

        seg_out_mask_path = os.path.join(server_root_path, "static", 'segmentation_result', 'mask', new_file_name)
        seg_out_img_path = os.path.join(server_root_path, "static", 'segmentation_result', 'mix', new_file_name)
        image_segmentation(upload_true_path, output_mask_path=seg_out_mask_path, output_img_path=seg_out_img_path)

        global detection_output_img_path_html, segmentaion_output_img_path_html, segmentaion_output_mask_path_html
        detection_output_img_path_html = detection_output_img_path[detection_output_img_path.find('detection_result'):]
        segmentaion_output_img_path_html = seg_out_img_path[seg_out_img_path.find('segmentation_result'):]
        segmentaion_output_mask_path_html = seg_out_mask_path[seg_out_mask_path.find('segmentation_result'):]

        return render_template('result.html',
                               query_path=query_path,
                               res_to_html=res_to_html)
    except:
        return redirect(url_for('error'))

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        save_request_file()
        return redirect(url_for('result'))
    else:
        return render_template('index.html')

@app.route('/editor', methods=['GET', 'POST'])
def editor():
    if request.method == 'POST':
        save_request_file()
        return redirect(url_for('result'))
    else:
        return render_template('editor.html')

@app.route('/about', methods=['GET', 'POST'])
def about():
    if request.method == 'POST':
        save_request_file()
        return redirect(url_for('result'))
    else:
        return render_template('about.html')

@app.route('/result', methods=['GET', 'POST'])
def result():
    if request.method == 'POST':
        save_request_file()
        return redirect(url_for('result'))
    else:
        return send_img_post_to_result_page()

@app.route('/result_detection', methods=['GET', 'POST'])
def result_detection():
    if request.method == 'POST':
        save_request_file()
        return redirect(url_for('result'))
    else:
        return render_template('result_detection.html', query_path=query_path,
                               detection_output_img_path=detection_output_img_path_html,
                               res_to_html=res_to_html)

@app.route('/result_segmentation', methods=['GET', 'POST'])
def result_segmentation():
    if request.method == 'POST':
        save_request_file()
        return redirect(url_for('result'))
    else:
        print(segmentaion_output_img_path_html, segmentaion_output_mask_path_html)
        return render_template('result_segmentation.html', query_path=query_path,
                               segmentaion_output_img_path_html=segmentaion_output_img_path_html,
                               segmentaion_output_mask_path_html=segmentaion_output_mask_path_html,
                               res_to_html=res_to_html)

@app.route('/error', methods=['GET', 'POST'])
def error():
    if request.method == 'POST':
        save_request_file()
        return redirect(url_for('result'))
    else:
        return render_template('error.html')

@app.errorhandler(404)
def page_not_found(error):
    return render_template('page_not_found.html'), 404

if __name__=="__main__":
    # app.debug = True
    app.run("0.0.0.0", port=5000)
    # http_server = WSGIServer(('', 5001), app)
    # http_server.serve_forever()
