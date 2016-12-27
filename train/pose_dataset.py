import random
import numpy as np
from PIL import Image
from chainer import dataset

## 2D/3D pose dataset
#
# The dataset of image and corresponding 2D/3D pose
class PoseDataset(dataset.DatasetMixin):
    ## constructor
    # @param path The filename of train/test file
    # @param mean The mean image
    # @param cropping_size The size of cropping for DNN training, the default value is (220x220)
    def __init__(self, path, mean, cropping_size=220):
        self.__mean = mean
        self.__cropping_size = cropping_size
        ## image filename
        self.__images = []
        ## the inverse matrix of scaled camera matrix
        self.__A_inv = []
        ## hstack of 2D pose (x0.T, x1.T, ...)
        self.__x_2d = []
        ## hstack of 3D pose (x0.T, x1.T, ...)
        self.__x_3d = []
        # read the file
        lines = []
        for line in open(path):
            lines.append(line[:-1])
        lines = lines[:-1]
        for line in lines:
            s = line.split(",")
            self.__images.append(s[0])
            s_f = map(float, s[1:])
            self.__A_inv.append(np.matrix([[s_f[0], s_f[1]], [s_f[2], s_f[3]], [s_f[4], s_f[5]]]))
            self.__x_2d.append(np.hstack(zip(s_f[6::5], s_f[7::5])))
            self.__x_3d.append(np.hstack(zip(s_f[8::5], s_f[9::5], s_f[10::5])))
    ## read image as numpy array
    # @param self The object pointer
    # @param path The path to image file
    # @return image array
    def __readImage(self, path):
        f = Image.open(path)
        try:
            image = np.asarray(f, dtype=np.float32)
        finally:
            f.close()
        return image.transpose(2, 0, 1)
    ## get length
    # @param self The object pointer
    # @return dataset length
    def __len__(self):
        return len(self.__images)
    ## get i-th example
    # @param self The object pointer
    # @param i The example index
    # @param i-th example
    def get_example(self, i):
        image = self.__readImage(self.__images[i])
        x_2d = self.__x_2d[i].copy()
        x_3d = self.__x_3d[i].copy()
        # data augumentation by random cropping
        _, h, w = image.shape
        top, left = random.randint(0, h - self.__cropping_size), random.randint(0, w - self.__cropping_size)
        bottom, right = top + self.__cropping_size, left + self.__cropping_size
        image = image[:, top:bottom, left:right]
        # modify 2D/3D pose according to the cropping
        for j in range(len(self.__x_2d[i])/2):
            du = np.array([left, top])
            x_2d[2*j:2*(j + 1)] -= du
            x_3d[3*j:3*(j + 1)] -= np.asarray(x_3d[3*j + 2]*np.matrix(du)*self.__A_inv[i].T)[0]
        # data augumentation by random noise
        C = np.cov(np.reshape(image, (3, -1)))
        l, e = np.linalg.eig(C)
        p = np.random.normal(0, 0.1)*np.matrix(e).T*np.sqrt(np.matrix(l)).T
        for c in range(3):
            image[i] += p[i]
        image = np.clip(image, 0, 255)
        # mean zeroing
        image -= self.__mean[:, top:bottom, left:right]
        # scale to [-1, 1]
        image /= 255.
        return image, x_2d, x_3d
