import sklearn
import cv2
import os
import numpy as np
from flask import Flask, request
import flask
import tensorflow as tf
from keras.models import load_model
from sklearn import cluster
app = Flask(__name__)


def loadmodel():
    print('start load model')
    global model
    model_dir = 'ad_prediction.h5' # model path
    if os.path.exists(model_dir):
        print('yes it is')
        model = load_model(model_dir)
    else:
        print('it doesnt')
    global graph
    graph = tf.get_default_graph()


def prepPNGimgs(array_of_image_paths):
    l = []
    for img_file in array_of_image_paths:  # for each file in the list of images...
        img = cv2.imread("{}".format(img_file))  # read the image...
        img = np.array(img, dtype=np.float64) / 255
        w, h, d = original_shape = tuple(img.shape)
        assert d == 3
        image_array = np.reshape(img, (w * h, d))
        kmeans = sklearn.cluster.KMeans(n_clusters=2, random_state=0).fit(image_array)
        labels = kmeans.predict(image_array)

        def recreate_image(codebook, labels, w, h):
            """Recreate the (compressed) image from the code book & labels"""
            d = codebook.shape[1]
            image = np.zeros((w, h, d))
            label_idx = 0
            for i in range(w):
                for j in range(h):
                    image[i][j] = (codebook[labels[label_idx]])
                    label_idx += 1
            return image

        clustImg = recreate_image(kmeans.cluster_centers_, labels, w, h)

        l.append(clustImg)

    return (np.asarray(l))


@app.route('/', methods=['GET', 'POST'])
@app.route('/index')
def index():
    return flask.render_template("render.html")

@app.route('/photo', methods=['GET', 'POST'])
def photo():
    data = dict()
    data['Success'] = True
    imglist = []
    img = request.files.get('file')
    print(img)
    path = "static/photo/"
    for i in os.listdir(path):
        path_file = os.path.join(path, i)
        if os.path.isfile(path_file):
            os.remove(path_file)
    file_path = os.path.join(path + img.filename)
    print(file_path)
    img.save(file_path)
    imglist.append(file_path)
    X = prepPNGimgs(imglist)
    print(X[0].shape)
    data['prediction'] = "AD"
    with graph.as_default():
        preds = model.predict(X).round()
    if preds[0][1] == 1:
        data["prediction"] = "Alzheimer’s Disease(AD)"
    else:
        data["prediction"] = "No Condition(NC)"
    

    return flask.jsonify(data)

print("start server")
loadmodel()
if __name__ == "__main__":
    app.run()