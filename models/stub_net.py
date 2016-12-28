import numpy as np
import chainer
import chainer.functions as F
import chainer.links as L

## The stub network
#
# The neural network for stub
class StubNet(chainer.Chain):
    ## constructor
    # @param Nj The number of joints
    def __init__(self, Nj):
        super(StubNet, self).__init__(
            l=L.Linear(None, Nj*3),
        )
        ## dropout flag (not used in class, it's for convnet)
        self.train = True
    ## predict 3D pose
    # @param self The object pointer
    # @param x The input image
    # @return predicted 3D pose
    def predict(self, x):
        return self.l(x)
    ## calculate loss function
    # @param self The object pointer
    # @param image The input image
    # @param x_2d The ground truth 2D joint positions
    # @param x_3d The ground truth 3D joint positions
    # @return loss
    def __call__(self, image, x_2d, x_3d):
        y = self.predict(image)
        loss = chainer.functions.mean_squared_error(y, x_3d)
        chainer.report({"loss": loss}, self)
        return loss
