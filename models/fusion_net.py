import numpy as np
import chainer
from chainer import cuda
import chainer.functions as F
import chainer.links as L

## The FusionNet
#
# The FusionNet : "Tekin, Bugra, et al. Fusing 2D Uncertainty and 3D Cues for Monocular Body Pose Estimation. arXiv preprint arXiv:1611.05708 (2016)."
class FusionNet(chainer.Chain):
    ## constructor
    # @param Nj The number of joints
    def __init__(self, Nj):
        super(FusionNet, self).__init__(
            conv11=L.Convolution2D(None, 32, 9, stride=1, pad=0),
            conv12=L.Convolution2D(None, 72, 5, stride=1, pad=1),
            conv13=L.Convolution2D(None, 72, 5, stride=1, pad=0),
            fc11  =L.Linear(None,  512),
            fc12  =L.Linear(None, 2048),
            conv21=L.Convolution2D(None, 32, 9, stride=1, pad=0),
            conv22=L.Convolution2D(None, 72, 5, stride=1, pad=1),
            conv23=L.Convolution2D(None, 72, 5, stride=1, pad=0),
            fc21  =L.Linear(None,  512),
            fc22  =L.Linear(None, 2048),
            fc1   =L.Linear(None, 1024),
            fc2   =L.Linear(None, 1024),
            fc3   =L.Linear(None, 3*Nj),
        )
        ## dropout flag
        self.train = True
    ## predict 2D joint location ditribution
    # @param self The object pointer
    # @param x The input image
    # @param x_2d The ground truth 2D joint position TODO:remove after U-Net works
    # @return predicted 2D joint location distribution
    def _predict2D(self, x, x_2d):
        # TODO:use U-Net
        xp = cuda.get_array_module(x)
        b, _, h, w = x.shape
        Nj = len(x_2d[0])/2
        X = np.zeros((b, Nj, h + 1, w + 1), dtype=np.float32)
        for i, x_2d_i in enumerate(x_2d):
            for j, x_2d_j in enumerate(F.split_axis(x_2d_i*h, Nj, 0)):
                # 4-nearest neighbor
                u0, v0 = x_2d_j[0].data, x_2d_j[1].data
                u = (int(xp.floor(u0)), int(xp.ceil(u0)))
                v = (int(xp.floor(v0)), int(xp.ceil(v0)))
                for u_k in u:
                    for v_l in v:
                        X[i, j, v_l, u_k] = xp.sqrt((u_k + 0.5 - u0)**2 + (v_l + 0.5 - v0)**2)
                X_sub = X[i, j, v[0]:v[1] + 1, u[0]:u[1] + 1]
                X_sub /= X_sub.sum()
        return X
    ## predict 3D pose
    # @param self The object pointer
    # @param x The input image
    # @param x_2d The ground truth 2D joint position TODO:remove after U-Net works
    # @return predicted 3D pose
    def predict(self, x, x_2d):
        # distribution map stream
        X = self._predict2D(x, x_2d)[:, :, 0:-1, 0:-1]
        h = F.relu(self.conv11(X))
        h = F.max_pooling_2d(h, 2, stride=2, pad=0)
        h = F.relu(self.conv12(h))
        h = F.max_pooling_2d(h, 2, stride=2, pad=0)
        h = F.relu(self.conv13(h))
        h = F.max_pooling_2d(h, 2, stride=2, pad=0)
        h = F.dropout(F.relu(self.fc11(h)), train=self.train)
        z_X = F.dropout(F.relu(self.fc12(h)), train=self.train)
        # image stream
        h = F.relu(self.conv21(x))
        h = F.max_pooling_2d(h, 2, stride=2, pad=0)
        h = F.relu(self.conv22(h))
        h = F.max_pooling_2d(h, 2, stride=2, pad=0)
        h = F.relu(self.conv23(h))
        h = F.max_pooling_2d(h, 2, stride=2, pad=0)
        h = F.dropout(F.relu(self.fc21(h)), train=self.train)
        z_I = F.dropout(F.relu(self.fc22(h)), train=self.train)
        # fusion network
        h = F.concat([z_X, z_I])
        h = F.dropout(F.relu(self.fc1(h)), train=self.train)
        h = F.dropout(F.relu(self.fc2(h)), train=self.train)
        y = self.fc3(h)
        return y
    ## calculate loss function
    # @param self The object pointer
    # @param image The input image
    # @param A The camera matrix
    # @param x_2d The ground truth 2D joint positions
    # @param x_3d The ground truth 3D joint positions
    # @return loss
    def __call__(self, image, A, x_2d, x_3d):
        y = self.predict(image, x_2d)
        loss = chainer.functions.mean_squared_error(y, x_3d)
        chainer.report({"loss": loss}, self)
        return loss
