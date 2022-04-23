from flask import Flask, render_template, request, redirect, flash, url_for,Markup
import main
import urllib.request
# from app import app
from werkzeug.utils import secure_filename
from model import getPrediction
import os
# import cv2
from PIL import Image

from flask import Flask

UPLOAD_FOLDER = './uploads'
STATIC_FOLDER = './static'

app = Flask(__name__)
app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['STATIC_FOLDER'] = STATIC_FOLDER

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/', methods=['POST'])
def submit_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No file selected for uploading')
            return redirect(request.url)
        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'],filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'],filename))
            # with Image.open(filepath) as im:
            #     rgb_im = im.convert('RGB')
            #     rgb_im.save(os.path.join('./static','myimg.jpg'))
            getPrediction(filename)
            label, time = getPrediction(filename)
            flash(label)
            flash(time)
            print(filepath)
            # img_tag = "<img src='{{ url_for"+'("static", filename='+"'"+filename+"'" +')}} alt="Food Img">'
            # flash(filepath)
            # print(img_tag)
            # flash(Markup(img_tag))
            return redirect('/')

@app.route('/effnet')
def effnet():
    return render_template('effnet.html')

@app.route('/resnet')
def resnet():
    return render_template('resnet.html')

@app.route('/onnx')
def onnx():
    return render_template('onnx.html')

@app.route('/stacking')
def stacking():
    return render_template('stacking.html')

if __name__ == "__main__":
    app.run()