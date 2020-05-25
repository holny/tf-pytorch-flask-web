import numpy as np
import tensorflow as tf
from flask import Flask, render_template, jsonify, request, make_response, send_from_directory, abort
from mnist import module as model

from cnocr import CnOcr
import mxnet as mx
from werkzeug.utils import secure_filename
import datetime
import random
import os
import base64

# import os


class Pic_str:
    def create_uuid(self): #生成唯一的图片的名称字符串，防止图片显示时的重名问题
        nowTime = datetime.datetime.now().strftime("%Y%m%d%H%M%S");  # 生成当前时间
        randomNum = random.randint(0, 100);  # 生成的随机整数n，其中0<=n<=100
        if randomNum <= 10:
            randomNum = str(0) + str(randomNum);
        uniqueNum = str(nowTime) + str(randomNum);
        return uniqueNum;



# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
# # 0.9表示可以使用GPU 90%的资源进行训练，可以任意修改
# sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
config = tf.ConfigProto(allow_soft_placement=True)

# 最多占gpu资源的70%
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)

# 开始不会给tensorflow全部gpu资源 而是按需增加
config.gpu_options.allow_growth = True

x = tf.placeholder("float", [None, 784])
sess = tf.Session(config=config)


with tf.variable_scope("regression"):
    print(model.regression(x))
    y1, variables = model.regression(x)
saver = tf.train.Saver(variables)
regression_file = tf.train.latest_checkpoint("mnist/data/regreesion.ckpt")
if regression_file is not None:
    saver.restore(sess, regression_file)

with tf.variable_scope("convolutional"):
    keep_prob = tf.placeholder("float")
    y2, variables = model.convolutional(x, keep_prob)
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver(variables)
convolutional_file = tf.train.latest_checkpoint(
    "mnist/data/convolutional.ckpt")
if convolutional_file is not None:
    saver.restore(sess, convolutional_file)


def regression(input):
    return sess.run(y1, feed_dict={x: input}).flatten().tolist()


def convolutional(input):
    return sess.run(
        y2, feed_dict={
            x: input,
            keep_prob: 1.0
        }).flatten().tolist()


app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False

UPLOAD_FOLDER = 'upload'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
basedir = os.path.abspath(os.path.dirname(__file__))
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'JPG', 'PNG', 'gif', 'GIF'])


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS


@app.route('/upload')
def upload_test():
    return render_template('up.html')


## 上传 文件 前后端已经联通(可以实现批量上传,只是前端禁用了批量上传, 这里逻辑是单个和批量上传都可以) by hly
# 上传文件
@app.route('/up_photo', methods=['POST'], strict_slashes=False)
def api_upload():
    file_dir = os.path.join(basedir, app.config['UPLOAD_FOLDER'])
    if not os.path.exists(file_dir):
        os.makedirs(file_dir)
    choose_ckpt = request.values.get("ckpt")
    print(choose_ckpt)
    f_list = request.files.getlist('photo')
    for f in f_list:
        print("filename:"+f.filename)
        if f and allowed_file(f.filename):
            fname = secure_filename(f.filename)
            print(fname)
            ext = fname.rsplit('.', 1)[1]
            new_filename = Pic_str().create_uuid() + '.' + ext
            f.save(os.path.join(file_dir, new_filename))

            # ocr = CnOcr()
            # img_fp = os.path.join(file_dir, new_filename)
            # img = mx.image.imread(img_fp, 1)
            # res = ocr.ocr(img)
            # print("Predicted Chars:", res)

            # return jsonify({"success": 0, "msg": res})
            # return jsonify({"success": 0, "msg": "success"})
            # return jsonify({"success": 0, "msg": "上传成功"})
        # else:
            # return jsonify({"error": 1001, "msg": "上传失败"})
    return jsonify({"success": 0, "msg": "success"})

@app.route('/download/<string:filename>', methods=['GET'])
def download(filename):
    if request.method == "GET":
        if os.path.isfile(os.path.join('upload', filename)):
            return send_from_directory('upload', filename, as_attachment=True)
        pass


# show photo
@app.route('/show/<string:filename>', methods=['GET'])
def show_photo(filename):
    file_dir = os.path.join(basedir, app.config['UPLOAD_FOLDER'])
    if request.method == 'GET':
        if filename is None:
            pass
        else:
            image_data = open(os.path.join(file_dir, '%s' % filename), "rb").read()
            response = make_response(image_data)
            response.headers['Content-Type'] = 'image/png'
            return response
    else:
        pass

# 路由
@app.route("/api/mnist", methods=['post'])
def mnist():

    input = ((255 - np.array(request.json, dtype=np.uint8)) / 255.0).reshape(
        1, 784)
    output1 = regression(input)
    output2 = convolutional(input)
    return jsonify(results=[output1, output2])


###############################################################################


# 上传训练数据集
# 已实现 批量上传数据 到指定 数据集
@app.route('/up_train_data', methods=['POST'], strict_slashes=False)
def api_upload_train_data():
    file_dir = os.path.join(basedir, app.config['UPLOAD_FOLDER'])
    # 上传同时，前端会发送数据源Id,根据不同Id，保存到不同数据源
    # dataSourceId=-1，说明是新建的数据源
    # **** 还  需要实现  保存已有数据源的信息，有哪些数据源，
    # 有了数据源后，新建训练就可以选择不同数据源来训练
    dataSourceName = request.values.get("dataSourceName")
    dataSourceId = request.values.get("dataSourceId")
    print('dataSourceName:'+dataSourceName)
    print('dataSourceId:' + dataSourceId)
    print('file_dir:'+file_dir)
    file_dir = os.path.join(file_dir,'dataSourceId',dataSourceId)
    if not os.path.exists(file_dir):
        os.makedirs(file_dir)

    f_list = request.files.getlist('upData')
    for f in f_list:
        print("filename:"+f.filename)
        if f and allowed_file(f.filename):
            fname = secure_filename(f.filename)
            print(fname)
            ext = fname.rsplit('.', 1)[1]
            new_filename = Pic_str().create_uuid() + '.' + ext
            f.save(os.path.join(file_dir, new_filename))

            # ocr = CnOcr()
            # img_fp = os.path.join(file_dir, new_filename)
            # img = mx.image.imread(img_fp, 1)
            # res = ocr.ocr(img)
            # print("Predicted Chars:", res)

            # return jsonify({"success": 0, "msg": res})
            # return jsonify({"success": 0, "msg": "success"})
            # return jsonify({"success": 0, "msg": "上传成功"})
        # else:
            # return jsonify({"error": 1001, "msg": "上传失败"})
    return jsonify({"success": 0, "msg": "success"})


## 开始训练
# 待完善
@app.route("/start_train", methods=['POST'])
def start_train():
    #  dataSourceId是必须存在的,  根据dataSourceId 得到训练的数据集Id
    #  而 ckptId可以没有(前端传来<0，说明不要ckpt)，不需要ckpt就是从新开始，如果有就是从这个ckpt开始训练。
    dataSourceId = request.values.get("dataSourceId")
    ckptId = request.values.get("ckptId")
    print("start_train-- dataSourceId:"+str(dataSourceId)+"  ckptId:"+str(ckptId))

    ## 模拟开始训练过程

    return jsonify({"success": 0, "msg": "接收到参数,开始训练!"})

## 前往 某个训练记录 详情页面
#待完善
@app.route("/record/<int:recordId>")
def record(recordId):
    ## recordId是 某个训练记录Id
    print("recordId:"+str(recordId))

    ## 前端显示 这个训练 的 状态 , 所用数据集, 总运行时间,训练进度，准确率, 每轮chart 等。  详见前端显示

    ## 模拟 生成 训练记录 详情

    return render_template("record.html")

# **********  下面只设置了 获取 dataSource list和 check point list route请求，
# ******   还未设置某个dataSource、某个data、某个ckpt的 CRUD 的route，  前端已经实现了crud效果，这里看情况是否添加

## 前端ajax传递获取 数据集列表请求
# 待完善
@app.route("/get_ds_list")
def get_dataSource_list():
    ## 模拟获取 数据集列表
    return jsonify({"success": 0, "msg": "DataSource List!"})

## 前往 某个数据集详情页面
#待完善
@app.route("/ds/<int:dataSourceId>")
def dataSource(dataSourceId):
    # 前端传递 dataSourceId 数据集 Id
    # 根据 dataSourceId , 往前端发送 这个数据集 的详情(图片和label列表)
    print("dataSourceId:"+str(dataSourceId))

    ## 模拟 生成 这个dataSourceId的数据集列表详情(图片和label列表)

    return render_template("data.html")

## 前端ajax传递获取 checkpoint 列表请求
# 待完善
@app.route("/get_ckpt_list")
def get_ckpt_list():
    ## 模拟获取 checkpoint 列表
    return jsonify({"success": 0, "msg": "checkpoint List!"})

## 前往 控制面板页面
# 待完善
@app.route("/dashboard")
def dashboard():
    ## 控制面板显示 总的 训练概览   详见前端显示

    return render_template("dashboard.html")

## 前往 生成训练页面
@app.route("/train")
def train():
    return render_template("train.html")

## 前往 数据集列表页面
@app.route("/datalist")
def datalist():
    return render_template("datalist.html")

#####################################################################################


@app.route("/")
def main():
    return render_template("index.html")


if __name__ == "__main__":
    app.debug = True
    app.run(host='0.0.0.0', port=8000)
