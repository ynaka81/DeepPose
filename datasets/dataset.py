import os
from itertools import chain
import numpy as np

## Dataset
#
# The base class of dataset
class Dataset(object):
    # define const values
    # TODO:define using joints
    JOINTS = ("torsoProximal", "torsoDistal", "upperLLegProximal", "upperLLegDistal", "lowerLLegProximal", "lowerLLegDistal", "upperRLegProximal", "upperRLegDistal", "lowerRLegProximal", "lowerRLegDistal", "upperLArmProximal", "upperLArmDistal", "lowerLArmProximal", "lowerLArmDistal", "upperRArmProximal", "upperRArmDistal", "lowerRArmProximal", "lowerRArmDistal", "headProximal", "headDistal")
    ## constructor
    # @param train_file The filename of train file, the default value is "train_data"
    # @param test_file The filename of test file, the default value is "test_data"
    def __init__(self, train_file="data/train_data", test_file="data/test_data"):
        try:
            os.makedirs(os.path.dirname(train_file))
            os.makedirs(os.path.dirname(test_file))
        except OSError:
            pass
        ## output file name
        self.__file = (open(train_file, "w"), open(test_file, "w"))
        ## train/test image filename (0:train, 1:test)
        self._images = [[], []]
        ## (A^-1*S)[:,0:2], A:camera matrix, S:scale diag matrix (3x2)
        self._A_inv = [[], []]
        ## 2D pose
        self._x_2d = [[], []]
        ## 3D pose
        self._x_3d = [[], []]
    ## save the dataset
    # @param self The object pointer
    # @param output The output file object
    # @param images The filenames of image
    # @param A_inv The inverse of scaled camera matrix
    # @param x_2d The 2D position of joints
    # @param x_3d The 3D position of joints
    def __save(self, output, images, A_inv, x_2d, x_3d):
        # check the size of arrays
        if not(len(images) == len(A_inv) == len(x_2d) == len(x_3d)):
            raise AttributeError("The length of attributes is not the same, images({0}), A_inv({1}), x_2d({2}) and x_3d({3}).".format(len(images), len(A_inv), len(x_2d), len(x_3d)))
        # output the file
        for images_i, A_inv_i, x_2d_t, x_3d_t  in zip(images, A_inv, x_2d, x_3d):
            x_i = np.vstack([np.vstack((x_2d_t[name], x_3d_t[name])) for name in self.JOINTS])
            output.write(",".join(map(str, [images_i] + list(chain.from_iterable(A_inv_i.tolist())) + list(chain.from_iterable(x_i.tolist())))) + os.linesep)
        output.write(os.linesep)
    ## save the train/test dataset
    # @param self The object pointer
    def _saveDataset(self):
        for i in (0, 1):
            self.__save(self.__file[i], self._images[i], self._A_inv[i], self._x_2d[i], self._x_3d[i])
