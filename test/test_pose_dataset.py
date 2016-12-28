import sys
import os
import numpy as np
from PIL import Image, ImageDraw
import unittest

sys.path.append("./")
from train.pose_dataset import PoseDataset

## Test PoseDataset
#
# The quantitative/qualitative tester for PoseDataset
class TestPoseDataset(unittest.TestCase):
    ## init test case
    # @param self The object pointer
    def setUp(self):
        ## pose dataset (0:for train, 1:for test)
        self.__dataset = [PoseDataset("test/train_data_for_test", np.load("data/mean.npy")), PoseDataset("test/train_data_for_test", np.load("data/mean.npy"), data_augmentation=False)]
        ## cropping size
        self.__cropping_size = 220
        ## joint length
        self.__N = 20
        ## the output directory name of qualitative test image
        self.__output_dirname = "test_result/PoseDataset"
        try:
            os.makedirs(self.__output_dirname)
        except OSError:
            pass
    ## test PoseDataset.__len__
    # @param self The object pointer
    def testDatasetLength(self):
        for dataset in self.__dataset:
            self.assertEqual(len(dataset), 4)
    ## test PoseDataset.get_example
    # @param self The object pointer
    def testGetExample(self):
        for id, dataset in enumerate(self.__dataset):
            N = len(dataset)
            for i in range(N):
                # repeat 10 times because get_example(i) function has randomness
                for j in range(10):
                    image, A, x_2d, x_3d = dataset.get_example(i)
                    # quantitative test for image and corresponding 2D/3D pose
                    self.assertEqual(image.shape, (3, self.__cropping_size, self.__cropping_size))
                    self.assertTrue((-1. < image).all())
                    self.assertTrue((image < 1.).all())
                    self.assertTrue((0 < x_2d).all())
                    self.assertTrue((x_2d < self.__cropping_size).all())
                    self.assertEqual(x_2d.shape, (2*self.__N,))
                    self.assertEqual(x_3d.shape, (3*self.__N,))
                    # qualitative test for image (scale to 0-255)
                    image_ = Image.fromarray(np.uint8((image.transpose(1, 2, 0) + 1)/2*255))
                    draw = ImageDraw.Draw(image_)
                    for k in range(len(x_2d)/2):
                        u, v = x_2d[2*k], x_2d[2*k + 1]
                        r = 2
                        draw.ellipse((u - r, v - r, u + r, v + r), fill="red")
                    image_.save(os.path.join(self.__output_dirname, "{0}_{1}_{2}.png".format(("train", "test")[id], i, j)))

if __name__ == "__main__":
    unittest.main()
