import sklearn
import cv2
import os
import numpy as np
from flask import Flask, request
import flask
import tensorflow as tf
from keras.models import load_model
from sklearn import cluster


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



loadmodel()
dirname = "OASIS_MR1_pngs/OAS1_0058_MR1"
imglist = os.listdir(dirname)
imglist = [os.path.join(dirname, i) for i in imglist]
X = prepPNGimgs(imglist)
preds = model.predict(X)
for i in range(preds.shape[0]):
    if preds[i][1] == 1:
        print("Alzheimer’s Disease(AD)")
    else:
        print("No Condition(NC)")