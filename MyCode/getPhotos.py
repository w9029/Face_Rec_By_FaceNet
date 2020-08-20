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

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

facenet_model_path = '../model/keras/model/facenet_keras.h5'
facenet_model = load_model(facenet_model_path)
cascade_path = '../model/cv2/haarcascade_frontalface_alt2.xml'
face_image_size = 160
image_dirpath = "../data/images/"
min_detect_size = 120

if __name__ == '__main__':

    a = UpdateFeatureInDB(facenet_model_path=facenet_model_path,
                          cascade_path=cascade_path,
                          face_image_size=face_image_size,
                          image_dirpath=image_dirpath)

    username = "yujunjie"
    path = image_dirpath+username
    isExists = os.path.exists(path)

    # 判断结果
    if not isExists:
        # 如果不存在则创建目录
        # 创建目录操作函数
        print("new user, create dic")
        os.makedirs(path)
    else:
        print("dic is already exist")

    cascade = cv2.CascadeClassifier(cascade_path)
    #cap=cv2.VideoCapture(0)
    cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)
    # 设置分辨率
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    count = 0
    while True:
        #从摄像头读取图片
        sucess,img=cap.read()

        faces = cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=3,
                                         minSize=(min_detect_size, min_detect_size))

        show_img = img.copy()
        for (x, y, w, h) in faces:
            cv2.rectangle(show_img, (x, y), (x + w, y + h), (0, 255, 0), 2)  # 画框

        cv2.imshow("img", show_img)

        #保持画面的持续。
        k=cv2.waitKey(1)
        if k == 27:
            #通过esc键退出摄像
            cv2.destroyAllWindows()
            break
        elif k==ord("s"):
            #通过s键保存图片
            cv2.imwrite(image_dirpath+"/"+username+"/{:0>5d}.jpg".format(count),img)
            count+=1
            #cv2.destroyAllWindows()
            #break
    #关闭摄像头
    cap.release()

    # 计算这个用户的特征
    if count >0:
        a.updateOneUserFeatures(username)
        a.updateOneUserAvgFeatures(username)

