import numpy as np
import os
import matplotlib.pyplot as plt
import cv2
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import proj3d
from imageio import imread
from skimage.transform import resize
from scipy.spatial import distance
from keras.models import load_model
import datetime
import time

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
from UpdataFeaturesInDB import UpdateFeatureInDB

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

def prewhiten(x):
    if x.ndim == 4:
        axis = (1, 2, 3)
        size = x[0].size
    elif x.ndim == 3:
        axis = (0, 1, 2)
        size = x.size
    else:
        raise ValueError('Dimension should be 3 or 4')

    mean = np.mean(x, axis=axis, keepdims=True)
    std = np.std(x, axis=axis, keepdims=True)
    std_adj = np.maximum(std, 1.0/np.sqrt(size))
    y = (x - mean) / std_adj
    return y

def l2_normalize(x, axis=-1, epsilon=1e-10):
    output = x / np.sqrt(np.maximum(np.sum(np.square(x), axis=axis, keepdims=True), epsilon))
    return output


def load_and_align_images(filepaths, margin):
    cascade = cv2.CascadeClassifier(cascade_path)

    aligned_images = []
    for filepath in filepaths:
        img = imread(filepath)

        faces = cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=3)

        (x, y, w, h) = faces[0]
        cropped = img[y - margin // 2:y + h + margin // 2,
                  x - margin // 2:x + w + margin // 2, :]
        aligned = resize(cropped, (face_image_size, face_image_size), mode='reflect')
        aligned_images.append(aligned)

    return np.array(aligned_images)

def resolve_an_img(img, user_avg_features, margin=10, batch_size=1):
    cascade = cv2.CascadeClassifier(cascade_path)
    aligned_images = []
    embs = []

    t2 = datetime.datetime.now()
    faces = cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=3,
                                     minSize=(min_detect_size, min_detect_size))
    t3 = datetime.datetime.now()
    print("cascade time is {}".format((t3 - t2).microseconds))

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)  # 画框

        cropped = img[y - margin // 2:y + h + margin // 2,
                  x - margin // 2:x + w + margin // 2, :]
        shape = np.shape(cropped)
        if(shape[0] == 0 or shape[1] == 0):
            print("Error! The pic has a 0 dim!")
            return embs

        aligned = resize(cropped, (face_image_size, face_image_size), mode='reflect')
        aligned = np.array(aligned)
        aligned = prewhiten(aligned)

        emb = facenet_model.predict_on_batch(aligned[np.newaxis])
        emb = l2_normalize(np.concatenate(emb))
        #print(np.shape(embs))

        # clac nearest face
        targetName = "unknown"
        targetDifference = 0.9
        for name in user_avg_features.keys():
            difference = float(distance.euclidean(emb, user_avg_features[name]))
            #print(difference)
            if(difference < targetDifference):
                targetDifference = difference
                targetName = name

        #txt = targetName + " {:.2f}".format(targetDifference)
        respercent = (targetDifference-0.3)/0.6 * 100
        if respercent > 100:
            respercent = 100
        elif respercent < 0:
            respercent = 0
        respercent = 100 - respercent
        txt = targetName + " {:.2f}%".format(respercent)

        # mark the name
        cv2.putText(img, txt, (x + 30, y + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)

    return embs

facenet_model_path = '../model/keras/model/facenet_keras.h5'
facenet_model = load_model(facenet_model_path)
cascade_path = '../model/cv2/haarcascade_frontalface_alt2.xml'
face_image_size = 160
image_dirpath = "../data/images/"
min_detect_size = 90

if __name__ == '__main__':

    a = UpdateFeatureInDB(facenet_model_path=facenet_model_path,
                          cascade_path=cascade_path,
                          face_image_size=face_image_size,
                          image_dirpath=image_dirpath)

    # wjh_f = a.dbManager.getUserFeatures("wjh")
    # wjh_avg_f = a.dbManager.getUserAvgFeature("wjh")
    # print(distance.euclidean(wjh_f[2],wjh_avg_f))

    user_avg_features = a.dbManager.getAllUserAndAvgFeature()
    print(user_avg_features.keys())

    #cap=cv2.VideoCapture(0)
    cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)
    # 设置分辨率
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1024)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 768)
    fps = 0.0
    while True:
        t1 = datetime.datetime.now()

        # 从摄像头读取图片
        sucess,img=cap.read()

        # clac embs and change img
        resolve_an_img(img, user_avg_features)

        cv2.putText(img, "fps: {:.1f}".format(fps), (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
        cv2.imshow("img", img)

        t2 = datetime.datetime.now()
        t = (t2-t1).microseconds
        print(str(t/1000.0)+"ms")
        print("************")
        fps = 1000000.0/t

        # 获取按键
        k=cv2.waitKey(1)
        if k == 27:
            #通过esc键退出摄像
            cv2.destroyAllWindows()
            break

    #关闭摄像头
    cap.release()

