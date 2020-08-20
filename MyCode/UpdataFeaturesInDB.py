import os
import sys
from DBManager import DBManager
import numpy as np
import matplotlib.pyplot as plt
import cv2
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import proj3d
from imageio import imread
from skimage.transform import resize
from skimage.transform import rescale
from scipy.spatial import distance
from keras.models import load_model

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

class UpdateFeatureInDB:
    def __init__(self, facenet_model_path, cascade_path, face_image_size, image_dirpath):

        self.facenet_model_path = facenet_model_path
        self.facenet_model = load_model(self.facenet_model_path)
        self.cascade_path = cascade_path
        self.face_image_size = face_image_size
        self.image_dirpath = image_dirpath

        # self.image_filepaths = [os.path.join(self.image_dirpath, f) for f in os.listdir(self.image_dirpath)]
        # print(self.image_filepaths)

        self.dbManager = DBManager()
        self.cascade = cv2.CascadeClassifier(self.cascade_path)

    #######################   avg features  #######################
    def updateAllUserAvgFeatures(self):
        names = self.dbManager.getAllUserName()
        for name in names:
            #print(name)
            self.updateOneUserAvgFeatures(name)


    def updateOneUserAvgFeatures(self, username):
        features = self.dbManager.getUserFeatures(username=username)
        avgFeature = []
        if features == False or len(features)  < 1:
            avgFeature = []
        else:
            avgFeature = np.average(features, axis=0)

        self.dbManager.setUserAvgFeature(username=username, avg_feature=avgFeature)

    #######################   features  ###########################
    def updataAllUserFeatures(self):
        for dic in os.listdir(self.image_dirpath):
            #print(dic)
            ret = self.updateOneUserFeatures(dic)
            #print(ret)
            if ret == False:
                break

    def updateOneUserFeatures(self, dic, margin=10):
        if not self.dbManager.isExistUser(dic):
            print("user {} is not exist, create".format(dic))
            self.dbManager.insertUser(username=dic, avg_feature=None)

        for file in os.listdir(os.path.join(self.image_dirpath, dic)):
            img = cv2.imread(os.path.join(self.image_dirpath, dic, file))
            img_width = img.shape[0]
            img_height = img.shape[1]
            a = img_width/600.0
            img_height = int(img_height/a)
            img_width = int(img_width/a)
            img = cv2.resize(img,dsize=(img_height, img_width))
            show_img = img.copy()
            faces = []
            faces = self.cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=3, minSize=(80,80))
            #faces = self.cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=3)

            i=0
            for (x, y, w, h) in faces:
                #print("w is {}".format(w))
                i+=1
                cv2.rectangle(show_img, (x, y), (x + w, y + h), (0, 255, 0), 2)  # 画框
                cv2.putText(show_img, str(i), (x + int(w/2), y + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)

            cv2.putText(show_img, "[{}]: {}".format(dic, file), (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            msg = "choose the face number"
            if len(faces) < 1:
                msg = "can not find people!"
                print("can not find faces in {}".format(file))
            cv2.putText(show_img, msg, (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.imshow("img", show_img)
            sys.stdin.flush()
            k = cv2.waitKey(0)

            if k == 27:
                # 通过esc键退出摄像
                cv2.destroyAllWindows()
                return False
                #break
            elif k >= ord("1") and k <= ord(str(len(faces))):
                # choose the face and clac emb
                print("choosed {}".format(k-48))
                flag = 1

                (x, y, w, h) = faces[k-48-1]
                cropped = img[y - margin // 2:y + h + margin // 2,
                          x - margin // 2:x + w + margin // 2, :]
                aligned = resize(cropped, (self.face_image_size, self.face_image_size), mode='reflect')
                aligned = np.array(aligned)
                aligned_image = self.prewhiten(aligned)

                emb = self.facenet_model.predict_on_batch(aligned_image[np.newaxis])
                emb = self.l2_normalize(np.concatenate(emb))

                if self.dbManager.isExistFeature(username=dic, filename=file):
                    self.dbManager.setFeature(username=dic, filename=file, feature=emb)
                else:
                    self.dbManager.insertFeature(username=dic, filename=file, feature=emb)

    ###########################  Util ###############################
    def prewhiten(self, x):
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

    def l2_normalize(self, x, axis=-1, epsilon=1e-10):
        output = x / np.sqrt(np.maximum(np.sum(np.square(x), axis=axis, keepdims=True), epsilon))
        #output = x / np.sqrt(np.max(np.sum(np.square(x), axis=axis, keepdims=True), epsilon))
        return output


if __name__ == '__main__':

    facenet_model_path = '../model/keras/model/facenet_keras.h5'
    cascade_path = '../model/cv2/haarcascade_frontalface_alt2.xml'
    face_image_size = 160
    image_dirpath = "../data/images/"

    a = UpdateFeatureInDB(facenet_model_path=facenet_model_path,
                          cascade_path=cascade_path,
                          face_image_size=face_image_size,
                          image_dirpath=image_dirpath)

    a.updataAllUserFeatures()
    #a.updateOneUserFeatures("test1")
    #a.updateAllUserAvgFeatures()

    wjh_f = a.dbManager.getUserFeatures("test1")
    wjh_avg_f = a.dbManager.getUserAvgFeature("wjh")

    #print(np.shape(wjh_f))
    print(distance.euclidean(wjh_f[0],wjh_avg_f))
    print(distance.euclidean(wjh_f[1],wjh_avg_f))
    print(distance.euclidean(wjh_f[2],wjh_avg_f))



