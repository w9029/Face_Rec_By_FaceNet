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

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)


model_path = '../model/keras/model/facenet_keras.h5'
model = load_model(model_path)

cascade_path = '../model/cv2/haarcascade_frontalface_alt2.xml'

image_dir_basepath = '../data/images/'
names = ['LarryPage', 'MarkZuckerberg', 'BillGates', 'wjh', 'wt', 'zjy', 'lkc']
image_size = 160

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

        faces = cascade.detectMultiScale(img,
                                         scaleFactor=1.1,
                                         minNeighbors=3)
        (x, y, w, h) = faces[0]
        cropped = img[y - margin // 2:y + h + margin // 2,
                  x - margin // 2:x + w + margin // 2, :]
        aligned = resize(cropped, (image_size, image_size), mode='reflect')
        aligned_images.append(aligned)

    return np.array(aligned_images)

def calc_embs(filepaths, margin=10, batch_size=1):
    aligned_images = prewhiten(load_and_align_images(filepaths, margin))
    pd = []
    for start in range(0, len(aligned_images), batch_size):
        pd.append(model.predict_on_batch(aligned_images[start:start+batch_size]))
    embs = l2_normalize(np.concatenate(pd))

    return embs

def calc_embs_byRAM(img, margin=10, batch_size=1):

    cascade = cv2.CascadeClassifier(cascade_path)

    aligned_images = []
    embs = []
    faces = cascade.detectMultiScale(img,
                                     scaleFactor=1.1,
                                     minNeighbors=3)

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)  # 画框

    if(len(faces) > 0):
        (x, y, w, h) = faces[0]
        print(x, y, w, h)
        if w <= 0 or h <= 0:
            print("warning!!!!!")
            return embs

        #cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)  # 画框

        # cv2.putText(frame, 'YourSelf',
        #             (x + 30, y + 30),
        #             cv2.FONT_HERSHEY_SIMPLEX,
        #             1,
        #             (255, 0, 255),
        #             2)  # 文字框，表示已经识别了本人

        cropped = img[y - margin // 2:y + h + margin // 2,
                  x - margin // 2:x + w + margin // 2, :]
        aligned = resize(cropped, (image_size, image_size), mode='reflect')
        aligned_images.append(aligned)

        aligned_images = np.array(aligned_images)

        aligned_images = prewhiten(aligned_images)

        pd = []
        for start in range(0, len(aligned_images), batch_size):
            pd.append(model.predict_on_batch(aligned_images[start:start+batch_size]))
        embs = l2_normalize(np.concatenate(pd))

    return embs

def calc_dist(img_name0, img_name1):
    return distance.euclidean(data[img_name0]['emb'], data[img_name1]['emb'])

def calc_dist_plot(img_name0, img_name1):
    print(calc_dist(img_name0, img_name1))
    plt.subplot(1, 2, 1)
    plt.imshow(imread(data[img_name0]['image_filepath']))
    plt.subplot(1, 2, 2)
    plt.imshow(imread(data[img_name1]['image_filepath']))
    plt.show()


data = {}
for name in names:
    image_dirpath = image_dir_basepath + name
    image_filepaths = [os.path.join(image_dirpath, f) for f in os.listdir(image_dirpath)]
    embs = calc_embs(image_filepaths)
    for i in range(len(image_filepaths)):
        data['{}{}'.format(name, i)] = {'image_filepath' : image_filepaths[i],
                                        'emb' : embs[i]}

#calc_dist_plot('BillGates0', 'LarryPage0')
# calc_dist_plot('wjh0', 'wjh1')

wjh_emb = data['wjh0']['emb'] + data['wjh1']['emb'] + data['wjh2']['emb'] + data['wjh3']['emb'] + data['wjh4']['emb'] + data['wjh5']['emb'] + data['wjh6']['emb']
print('wjh_emb_all is {}'.format(wjh_emb))
print('wjh_emb_0 is {}'.format(data['wjh0']['emb']))
wjh_emb/= 7
print('wjh_emb_ave is {}'.format(wjh_emb))

cap=cv2.VideoCapture(0)
while True:
    #从摄像头读取图片
    sucess,img=cap.read()
    #gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    embs=[]
    t2 = datetime.datetime.now()
    embs = calc_embs_byRAM(img)
    t3 = datetime.datetime.now()

    #print(len(embs))
    t5 = t4 =0
    if(len(embs) > 0):
        #value = distance.euclidean(embs[0], data['wjh1']['emb'])
        value = distance.euclidean(embs[0], wjh_emb)
        print(value)

    print((t3-t2).microseconds)
    print("************")

    cv2.imshow("img", img)

    #保持画面的持续。
    k=cv2.waitKey(1)
    if k == 27:
        #通过esc键退出摄像
        cv2.destroyAllWindows()
        break
    elif k==ord("s"):
        #通过s键保存图片，并退出。
        cv2.imwrite("image2.jpg",img)
        cv2.destroyAllWindows()
        break
#关闭摄像头
cap.release()

