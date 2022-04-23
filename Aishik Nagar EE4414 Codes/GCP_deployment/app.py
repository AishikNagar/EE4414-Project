from flask import Flask, render_template, request, redirect, flash, url_for,Markup
import main
import urllib.request
# from app import app
from werkzeug.utils import secure_filename
from model import getPrediction
import os
# import cv2

from flask import Flask

UPLOAD_FOLDER = './uploads'

app = Flask(__name__)
app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

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
            # file.save(os.path.join(app.config['UPLOAD_FOLDER'],'uploaded_img.jpg'))
            file.save(os.path.join('./static','uploaded_img.jpg'))
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


if __name__ == "__main__":
    app.run()