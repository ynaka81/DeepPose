import numpy
import six

import chainer
from chainer import cuda
from chainer import function
from chainer.utils import type_check

class ChannelWiseSoftmaxCrossEntropy(function.Function):
    ## constructor
    # @param cache_score When it is True, the function stores result of forward computation to use it on backward computation
    def __init__(self, cache_score=True):
        self.cache_score = cache_score
    ## check type
    # @param self The object pointer
    # @param in_types The input variables
    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 2)
        x_type, t_type = in_types
        type_check.expect(
            x_type.dtype.kind == "f",
            t_type.dtype.kind == "f",
            x_type.ndim > 2,
            t_type.ndim == x_type.ndim,
            x_type.shape == t_type.shape,
        )
    ## check input variables
    # @param self The object pointer
    # @param x The output of the last layer of network
    # @param t The ground truth distribution of 2D position
    def _check_input_values(self, x, t):
        xp = cuda.get_array_module(x)
        d = 1.e-10
        s = xp.maximum(t, 0).sum(axis=tuple(range(2, n)))
        if not(t <= 1 and (1 - d < s).all() and (s < 1 + d).all()):
            raise ValueError("Each label 't' need to satisfy 't <= 1' and 'sum max(t, 0) == 1'.")
    ## computes channel-wise softmax and return it's components
    # @param self The object pointer
    # @param x The input variable
    # @return softmax, L, log sum exp(L)
    def _computeSoftmax(self, x):
        xp = cuda.get_array_module(x)
        axis = tuple(range(2, x.ndim))
        m = x.max(axis=axis, keepdims=True)
        y = x - m
        xp.exp(y, out=y)
        s = y.sum(axis=axis, keepdims=True)
        xp.log(s, out=s)
        return x - m - s, x - m, s
    # @param self The object pointer
    # @param inputs The input variables
    # @return The cross entropy loss
    def forward(self, inputs):
        xp = cuda.get_array_module(*inputs)
        x, t = inputs
        axis = tuple(range(2, x.ndim))
        # check input variables
        if chainer.is_debug():
            self._check_input_values(x, t)
        # compute channel-seize softmax
        log_X, L, sum_L = self._computeSoftmax(x)
        if self.cache_score:
            self.X = xp.exp(log_X)
        # compute cross entropy loss
        self._coeff = 1./reduce(lambda x, y: x*y, x.shape[:2])
        L = -self._coeff*((xp.maximum(t, 0)*L).sum(axis=axis, keepdims=True) - sum_L).sum(keepdims=True)
        return L.reshape(()),
    ## compute backward calculation
    # @param self The object pointer
    # @param inputs The input variables
    # @param grad_outputs The output gradients
    # @return The backward calculation of cross entropy loss
    def backward(self, inputs, grad_outputs):
        xp = cuda.get_array_module(*inputs)
        x, t = inputs
        gl = grad_outputs[0]
        if hasattr(self, 'y'):
            X = self.X.copy()
        else:
            log_X, _, _ = self._computeSoftmax(x)
            X = xp.exp(log_X)
        gx = -self._coeff*(xp.maximum(t, 0) - X)*gl
        return gx, None

## computes cross entropy loss for channel-wise pre-softmax activations
# @param x The output of the last layer of network
# @param t The ground truth distribution of 2D position, if t[u, v] < 0, corresponding x[u, v] is ignored
# @param cache_score When it is True, the function stores result of forward computation to use it on backward computation
# @return The cross entropy loss
def channel_wise_softmax_cross_entropy(x, t, cache_score=True):
    return ChannelWiseSoftmaxCrossEntropy(cache_score)(x, t)
