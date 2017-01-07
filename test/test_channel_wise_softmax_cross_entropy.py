import sys
import unittest
import numpy as np
from chainer import gradient_check

sys.path.append("./")
from models.functions.channel_wise_softmax_cross_entropy import ChannelWiseSoftmaxCrossEntropy

## Test ChannelWiseSoftmaxCrossEntropy
#
# The quantitative tester for ChannelWiseSoftmaxCrossEntropy
class TestChannelWiseSoftmaxCrossEntropy(unittest.TestCase):
    ## init test case
    # @param self The object pointer
    def setUp(self):
        ## channel wise softmax cross entropy function
        self.F = ChannelWiseSoftmaxCrossEntropy()
    ## create test variables
    # @param self The object pointer
    # @return x, t
    def __createTestVariable(self):
        # const values
        batch = 2
        crop = 10
        Nj = 5
        # create test variables
        x = np.random.rand(batch, Nj, crop, crop)
        t = 2*np.random.rand(batch, Nj, crop, crop) - 1
        t /= np.maximum(t, 0).sum(axis=(2, 3), keepdims=True)
        return x, t
    ## test channel-wise softmax
    # @param self The object pointer
    def testComputeSoftmax(self):
        for i in range(10):
            x, t = self.__createTestVariable()
            log_y, _, _ = self.F._computeSoftmax(x)
            y = np.exp(log_y)
            d = 1.e-10
            y_sum = y.sum(axis=(2, 3))
            self.assertTrue((1 - d < y_sum).all())
            self.assertTrue((y_sum < 1. + d).all())
    ## test forward
    # @param self The object pointer
    def testForward(self):
        for i in range(10):
            x, t = self.__createTestVariable()
            y = self.F.forward((x, t))
            self.assertEqual(len(y), 1)
            self.assertEqual(type(y[0]), np.ndarray)
            self.assertEqual(y[0].ndim, 0)
            self.assertTrue((y[0] >= 0).all())
    ## test backward
    # @param self The object pointer
    def testBackward(self):
        for i in range(10):
            gradient_check.check_backward(self.F, self.__createTestVariable(), None, no_grads=(False, True))

if __name__ == "__main__":
    unittest.main()
