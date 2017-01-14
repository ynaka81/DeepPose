import os
import imp
import numpy as np
import chainer

## 3D pose estimator
#
# The 3D pose estimator by using DeepNet
class Deep3dPose(object):
    ## constructor
    # @param model Model definition file in models dir
    # @param Nj Number of joints
    def __init__(self, model, Nj):
        module_name  = os.path.basename(model).split(".")[0]
        class_name = "".join([s.capitalize() for s in module_name.split("_")])
        _model = imp.load_source(module_name, model)
        _model = getattr(_model, class_name)
        ## model class
        self.model = _model(Nj)
        self.model.train = False
        ## number of joints
        self.__Nj = Nj
    ## initialize model with given file
    # @param self The object pointer
    # @param model_file The trained model parameter file
    def init(self, model_file):
        chainer.serializers.load_npz(model_file, self.model)
    ## estimate 3D pose
    # @param self The object pointer
    # @param image The input image
    # @return 3D pose
    def __call__(self, image):
        y = self.model.predict(np.array([image]))
        p = []
        for i in range(self.__Nj):
            p.append(np.matrix(y.data[:, 3*i:3*(i + 1)]).T)
        return p
