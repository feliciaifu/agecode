from age_and_gender import *
from PIL import Image, ImageDraw, ImageFont
import face_recognition
import numpy
import os
import pprint

import imageio
import numpy as np
import tensorflow as tf
from skimage.transform import resize

from detectface import create_mtcnn, detect_face
data = AgeAndGender()
data.load_shape_predictor('models/shape_predictor_5_face_landmarks.dat')
data.load_dnn_gender_classifier('models/dnn_gender_classifier_v1.dat')
data.load_dnn_age_predictor('models/dnn_age_predictor_v1.dat')

gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
output_dir = "alignface/output_dir"
with tf.Graph().as_default():
    sess = tf.compat.v1.Session()
    with sess.as_default():
        pnet, rnet, onet = create_mtcnn(sess, None)


def alignFace(int_name: str, out_name: str):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # 脸的最小大小
    minFaceSize = 15
    # 人脸检测有三部，每一步阈值如下
    threshold = [0.55, 0.65, 0.75]
    # 缩放因子，用来构造图像金字塔
    factor = 0.825
    try:
        img = imageio.imread(int_name)
    except(IOError, ValueError, IndexError) as ex:
        print(ex)
    else:
        img = img[:, :, 0:3]
        # 找出人脸框
        boxs, _ = detect_face(img, minFaceSize, pnet, rnet, onet, threshold, factor)

        # 找到多少个人脸，应为一个
        face_nums = boxs.shape[0]
        if face_nums == 1:
            detected_face = boxs[:, 0:4]

            # result = data.predict(img, detected_face)
            img_size = np.asarray(img.shape)[0:2]
            pprint.pp(img_size)
            detected_face = np.squeeze(detected_face)
            pprint.pprint(detected_face)
            tarbox = np.zeros(4, dtype=np.int32)
            tarbox[0] = np.maximum(detected_face[0] - 44 / 2, 0)
            tarbox[1] = np.maximum(detected_face[1] - 44 / 2, 0)
            tarbox[2] = np.minimum(detected_face[2] + 44 / 2, img_size[1])
            tarbox[3] = np.minimum(detected_face[3] + 44 / 2, img_size[0])
            # 已找出的人脸
            cropped = img[tarbox[1]:tarbox[3], tarbox[0]:tarbox[2], :]
            scaled = resize(cropped, output_shape=(112, 96))
            imageio.imsave(out_name, scaled)
            return True
        else:
            imageio.imsave(out_name, img)
            return False

def guess(dir_name,dir_out,txt_path):
    abpos = -1
    ress = open(txt_path, "w")
    if os.path.isdir(dir_name):
        for relpath in os.listdir(dir_name):
            abspath = os.path.join(dir_name, relpath)
            abpos = abpos + 1
            photopos = 0
            if not os.path.exists(dir_out + "/" + str(abpos) + "/"):
                os.mkdir(dir_out + "/" + str(abpos) + "/")
            for relpath1 in os.listdir(abspath):
                abspath1 = os.path.join(abspath, relpath1)
                if os.path.isfile(abspath1) and any(
                        [abspath1.endswith('.' + ty) for ty in ('jpg', 'png', 'JPG', 'PNG', 'jpeg')]):
                    img = Image.open(abspath1).convert("RGB")
                    face_bounding_boxes = face_recognition.face_locations(
                        numpy.asarray(img),  # Convert to numpy array
                        model='hog'  # 'hog' for CPU | 'cnn' for GPU (NVIDIA with CUDA)
                    )
                    if len(face_bounding_boxes) == 0:
                        if not alignFace(abspath1, "result.jpg"):
                            ress.writelines(
                                "Face_recognition Failure,MTCNN Failure" + abspath1 + "   " + dir_out + "/" + str(
                                    abpos) + "/" + str(photopos) + "\n")
                        else:
                            img = Image.open("result.jpg").convert("RGB")
                            result = data.predict(img)
                            if len(result) != 0:
                                outfile = dir_out + "/" + str(abpos) + "/" + str(photopos) + "_" + str(
                                    result[0]['age']['value']) + ".jpg"
                                photopos = photopos + 1
                                print(outfile)
                                img.resize((96, 116), Image.ANTIALIAS).save(outfile)
                            else:
                                ress.writelines(
                                    "Face_recognition Failure,MTCNN success,Guess Failure" + abspath1 + "   " + dir_out + "/" + str(
                                        abpos) + "/" + str(photopos) + "\n")
                    else:
                        result = data.predict(img, face_bounding_boxes)
                        if len(result) != 0:
                            outfile = dir_out + "/" + str(abpos) + "/" + str(photopos) + "_" + str(
                                result[0]['age']['value']) + ".jpg"
                            photopos = photopos + 1
                            detected_face = np.squeeze(face_bounding_boxes[0])
                            img = imageio.imread(abspath1)
                            img = img[:, :, 0:3]
                            cropped = img[detected_face[0]:detected_face[2], detected_face[3]:detected_face[1], :]
                            scaled = resize(cropped, output_shape=(112, 96))
                            print(outfile)
                            imageio.imsave(outfile, scaled)
                        else:
                            ress.writelines(
                                "Face_recognition Failure" + abspath1 + "   " + dir_out + "/" + str(abpos) + "/" + str(
                                    photopos) + "\n")
    ress.close()

if __name__=='__main__':
    guess("/home/yifu/下载/CACD_VS_OUT_OUT","/home/yifu/下载/CACD_VS_RES","cacd.txt")
