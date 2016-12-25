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
        self.__file = (open(train_file, "w"), open(test_file, "w"))
        self._images = [[], []]
        self._P_ = [[], []]
        self._x_2d = [[], []]
        self._x_3d = [[], []]
    ## save the dataset
    # @param self The object pointer
    # @param output The output file object
    # @param images The filenames of image
    # @param P_ The pseudo inverse of camera matrix
    # @param x_2d The 2D position of joints
    # @param x_3d The 3D position of joints
    def __save(self, output, images, P_, x_2d, x_3d):
        # check the size of arrays
        if not(len(images) == len(P_) == len(x_2d) == len(x_3d)):
            raise AttributeError("The length of attributes is not the same, images({0}), P_({1}), x_2d({2}) and x_3d({3}).".format(len(images), len(P_), len(x_2d), len(x_3d)))
        # output the file
        for images_i, P_i, x_2d_t, x_3d_t  in zip(images, P_, x_2d, x_3d):
            x_i = np.vstack([np.vstack((x_2d_t[name], x_3d_t[name])) for name in self.JOINTS])
            output.write(",".join(map(str, [images_i] + list(chain.from_iterable(P_i.tolist())) + list(chain.from_iterable(x_i.tolist())))))
    ## save the train/test dataset
    # @param self The object pointer
    def _saveDataset(self):
        for i in (0, 1):
            self.__save(self.__file[i], self._images[i], self._P_[i], self._x_2d[i], self._x_3d[i])
