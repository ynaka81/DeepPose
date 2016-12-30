import numpy as np
import chainer
import chainer.functions as F
import chainer.links as L

## The AlexNet
#
# The AlexNet : "A. Krizhevsky, I. Sutskever, and G. Hinton.  Imagenet clas-sification with deep convolutional neural networks. InNIPS , 2012"
class AlexNet(chainer.Chain):
    ## constructor
    # @param Nj The number of joints
    def __init__(self, Nj):
        super(AlexNet, self).__init__(
            conv1=L.Convolution2D(None,  96, 11, stride=4),
            conv2=L.Convolution2D(None, 256,  5, pad=2),
            conv3=L.Convolution2D(None, 384,  3, pad=1),
            conv4=L.Convolution2D(None, 384,  3, pad=1),
            conv5=L.Convolution2D(None, 256,  3, pad=1),
            fc6=L.Linear(None, 4096),
            fc7=L.Linear(None, 4096),
            fc8=L.Linear(None, Nj*3),
        )
        ## dropout flag
        self.train = True
    ## predict 3D pose
    # @param self The object pointer
    # @param x The input image
    # @return predicted 3D pose
    def predict(self, x):
        # layer1
        h = F.relu(self.conv1(x))
        h = F.local_response_normalization(h)
        h = F.max_pooling_2d(h, 3, stride=2)
        # layer2
        h = F.relu(self.conv2(h))
        h = F.local_response_normalization(h)
        h = F.max_pooling_2d(h, 3, stride=2)
        # layer3-5
        h = F.relu(self.conv3(h))
        h = F.relu(self.conv4(h))
        h = F.relu(self.conv5(h))
        h = F.max_pooling_2d(h, 3, stride=2)
        # layer6-8
        h = F.dropout(F.relu(self.fc6(h)), train=self.train)
        h = F.dropout(F.relu(self.fc7(h)), train=self.train)
        h = self.fc8(h)
        return h
    ## calculate loss function
    # @param self The object pointer
    # @param image The input image
    # @param A The camera matrix
    # @param x_2d The ground truth 2D joint positions
    # @param x_3d The ground truth 3D joint positions
    # @return loss
    def __call__(self, image, A, x_2d, x_3d):
        y = self.predict(image)
        loss = chainer.functions.mean_squared_error(y, x_3d)
        chainer.report({"loss": loss}, self)
        return loss
