import numpy as np
import chainer
from chainer import cuda
import chainer.functions as F
import chainer.links as L

from models.functions.channel_wise_softmax_cross_entropy import channel_wise_softmax_cross_entropy

## The U-Net
#
# The U-Net : "O.  Ronneberger,  P.  Fischer,  and  T.  Brox.   U-Net:  Convo-lutional Networks for Biomedical Image Segmentation.   In MICCAI, 2015"
class UNet(chainer.Chain):
    ## constructor
    # @param Nj The number of joints
    def __init__(self, Nj):
        super(UNet, self).__init__(
            conv1 =L.Convolution2D(None,  128, 3, stride=1,  pad=0),
            conv2 =L.Convolution2D(None,  256, 3, stride=1,  pad=0),
            conv3 =L.Convolution2D(None,  512, 3, stride=1,  pad=0),
            conv4 =L.Convolution2D(None, 1024, 3, stride=1,  pad=0),
            conv5 =L.Convolution2D(None, 2048, 3, stride=1,  pad=1),
            conv6 =L.Convolution2D(None, 1024, 3, stride=1,  pad=1),
            conv7 =L.Convolution2D(None,  512, 3, stride=1,  pad=1),
            conv8 =L.Convolution2D(None,  256, 3, stride=1,  pad=1),
            conv9 =L.Convolution2D(None,  128, 3, stride=1,  pad=1),
            conv10=L.Convolution2D(None,   Nj, 1, stride=1, pad=10),
            upconv1=L.Deconvolution2D(None, 1024, 2, stride=2, pad=0),
            upconv2=L.Deconvolution2D(None,  512, 2, stride=2, pad=0),
            upconv3=L.Deconvolution2D(None,  256, 2, stride=2, pad=0),
            upconv4=L.Deconvolution2D(None,  128, 2, stride=2, pad=0),
        )
        ## dropout flag (not used in class, it's for convnet)
        self.train = True
    ## predict 2D pose distribution
    # @param self The object pointer
    # @param x The input image
    # @return predicted 2D pose distribution
    def predict(self, x):
        xp = cuda.get_array_module(x.data)
        # contracting path
        c1 = F.relu(self.conv1(x))
        c2 = F.max_pooling_2d(c1, 2, stride=2, pad=0)
        c3 = F.relu(self.conv2(c2))
        c4 = F.max_pooling_2d(c3, 2, stride=2, pad=0)
        c5 = F.relu(self.conv3(c4))
        c6 = F.max_pooling_2d(c5, 2, stride=2, pad=0)
        c7 = F.relu(self.conv4(c6))
        c8 = F.max_pooling_2d(c7, 2, stride=2, pad=0)
        # concat
        s  = c7.shape
        z  = xp.zeros([s[0], s[1], s[2], 1], dtype=np.float32)
        m1 = F.concat([c7, z], axis=3)
        z  = xp.zeros([s[0], s[1], 1, s[3] + 1], dtype=np.float32)
        m1 = F.concat([m1, z], axis=2)
        # expansive path
        e1 = F.relu(self.conv5(c8))
        e2 = self.upconv1(e1)
        e3 = self.conv6(F.concat([m1, e2]))
        e4 = self.upconv2(e3)
        e5 = self.conv7(F.concat([c5[:, :, 1:53, 1:53], e4]))
        e6 = self.upconv3(e5)
        e7 = self.conv8(F.concat([c3[:, :, 3:107, 3:107], e6]))
        e8 = self.upconv4(e7)
        e9 = self.conv9(F.concat([c1[:, :, 8:216, 8:216], e8]))
        return self.conv10(e9)
    ## calculate loss function
    # @param self The object pointer
    # @param image The input image
    # @param A The camera matrix
    # @param x_2d The ground truth 2D joint positions
    # @param x_3d The ground truth 3D joint positions
    # @return loss
    def __call__(self, image, A, x_2d, x_3d):
        y = self.predict(image)
        xp = cuda.get_array_module(y.data)
        # generate ground truth labels
        _, Nj, h, w = y.shape
        t = xp.full(y.shape, -1, dtype=np.float32)
        for i, x_2d_i in enumerate(x_2d):
            for j, x_2d_j in enumerate(F.split_axis(x_2d_i*h, Nj, 0)):
                # 4-nearest neighbor
                u0, v0 = x_2d_j[0].data, x_2d_j[1].data
                u = (int(xp.floor(u0)), int(xp.ceil(u0)))
                v = (int(xp.floor(v0)), int(xp.ceil(v0)))
                for u_k in u:
                    for v_l in v:
                        t[i, j, v_l, u_k] = xp.sqrt((u_k + 0.5 - u0)**2 + (v_l + 0.5 - v0)**2)
                t_sub = t[i, j, v[0]:v[1] + 1, u[0]:u[1] + 1]
                t_sub /= t_sub.sum()
        # calculate loss
        loss = channel_wise_softmax_cross_entropy(y, t)
        chainer.report({"loss": loss}, self)
        return loss
