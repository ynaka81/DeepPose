import numpy as np
import cv2

## Bounding box of image
#
# The bounding box of the image calculated by the given 2D human pose
class BoundingBox(object):
    ## constructor
    # @param image The base image
    # @param x_2d The 2d pose of human
    # @param image_size The size of output image, the default value is (256x256)
    # @param cropping_size The size of cropping for DNN training, the default value is (220x220)
    # @param margin The cropping margin, default value is 10
    def __init__(self, image, x_2d, image_size=256, cropping_size=220, margin=10):
        if 2*(cropping_size - margin) - image_size <= 0:
            raise ValueError("Bad image and cropping size, 2 x (cropping_size({0}) - margin({1})) - image_size({2}) should be > 0".format(cropping_size, margin, image_size))
        # calculate bounding box
        x_2d_list = x_2d.get()[1]
        x = [float(v[0]) for v in x_2d_list]
        y = [float(v[1]) for v in x_2d_list]
        x_min, x_max = min(x), max(x)
        y_min, y_max = min(y), max(y)
        # extend bounding box considering cropping size
        c = cropping_size - margin
        k = 1.0*max(x_max - x_min, y_max - y_min)/(2*c - image_size)
        h, w, _ = image.shape
        if (x_max - x_min) > (y_max - y_min):
            u_min = max(x_max - k*c, 0)
            u_max = min(x_min + k*c, w)
            w_2 = (u_max - u_min)/2
            y_c = (y_max + y_min)/2
            v_min = max(y_c - w_2, 0)
            v_max = min(y_c + w_2, h)
        else:
            v_min = max(y_max - k*c, 0)
            v_max = min(y_min + k*c, h)
            h_2 = (v_max - v_min)/2
            x_c = (x_max + x_min)/2
            u_min = max(x_c - h_2, 0)
            u_max = min(x_c + h_2, w)
        u_min, u_max, v_min, v_max = int(np.floor(u_min)), int(np.ceil(u_max)), int(np.floor(v_min)), int(np.ceil(v_max))
        cropping_image = image[v_min:v_max, u_min:u_max]
        ## the image cropped by bounding box
        self.image = cv2.resize(cropping_image, (image_size, image_size))
        ## bounding box image coordinate origin
        self.u_0 = np.matrix([u_min, v_min, 0.]).T
        ## scaling factor
        self.s = 1./k
