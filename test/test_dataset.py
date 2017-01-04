import os
import random
import unittest
import cv2
import numpy as np

## Test dataset quality
#
# The quantitative/qualitative tester for dataset
class TestDataset(unittest.TestCase):
    ## init test case
    # @param self The object pointer
    def setUp(self):
        # load dataset contents
        lines = []
        for line in open("data/train_data"):
            lines.append(line[:-1])
        self.__lines = lines[:-1]
        ## size of cropping for DNN training
        self.__cropping_size = 227
        ## joint length
        self.__Nj = 14
        ## output directory name of qualitative test image
        self.__output_dirname = "test_result/Dataset"
        try:
            os.makedirs(self.__output_dirname)
        except OSError:
            pass
    ## test data length
    # @param self The object pointer
    def testDataLength(self):
        N = 1 + 9 + 5*self.__Nj
        for line in self.__lines:
            self.assertEqual(len(line.split(",")), N)
    ## test image existence
    # @param self The object pointer
    def testImageExist(self):
        for line in self.__lines:
            image = line.split(",")[0]
            self.assertTrue(os.path.isfile(image))
    ## test camera matrix
    # @param self The object pointer
    def testCameraMatrix(self):
        for line in self.__lines:
            s = map(np.float32, line.split(",")[1:])
            self.assertEqual(s[8], 1)
            for i, s_i in enumerate(s[:9]):
                if i in (0, 2, 4, 5, 8):
                    self.assertNotEqual(s_i, 0)
                else:
                    self.assertEqual(s_i, 0)
    ## test 2D position
    # @param self The object pointer
    def test2dPosition(self):
        # for random test
        samples = np.random.randint(len(self.__lines), size=10)
        # crawl dataset
        margin = 10
        for i, line in enumerate(self.__lines):
            split = line.split(",")
            image = cv2.imread(split[0])
            h, w, _ = image.shape
            w_, h_ = w - self.__cropping_size, h - self.__cropping_size
            s = map(np.float32, split[1 + 9:])
            x = np.hstack(zip(s[0::5], s[1::5]))
            # quantitative test
            self.assertTrue((np.floor(x + margin) <= (self.__cropping_size, self.__cropping_size)*self.__Nj).all())
            self.assertTrue(((w_, h_)*self.__Nj <= np.ceil(x - margin)).all())
            # qualitative test, random 10 sample
            if i in samples:
                # test image and corresponding 2D pose
                for j, x_j in enumerate(np.hsplit(x, self.__Nj)):
                    cv2.circle(image, tuple(x_j), 3, (0, 0, 255), -1)
                    cv2.putText(image, str(j), tuple(x_j), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255))
                # cropping test
                for j, u_0 in enumerate(((0, 0), (w_, 0), (0, h_), (w_, h_))):
                    cv2.imwrite(os.path.join(self.__output_dirname, "{0}_{1}.png".format(np.where(samples == i)[0][0], j)), image[u_0[0]:u_0[0] + self.__cropping_size, u_0[1]:u_0[1] + self.__cropping_size])
    ## test 3D position
    # @param self The object pointer
    def test3dPosition(self):
        for line in self.__lines:
            split = line.split(",")
            s = map(np.float32, split[1:])
            A = np.matrix([[s[0], s[1], s[2]], [s[3], s[4], s[5]], [s[6], s[7], s[8]]])
            x_2d = np.matrix(np.hstack(zip(s[9::5], s[10::5])))
            x_3d = np.matrix(np.hstack(zip(s[11::5], s[12::5], s[13::5])))
            # 2D->3D mapping error should be ignorable because of nonlinear factor
            for i in range(self.__Nj):
                x = x_3d[:, 3*i:3*(i + 1)]*A.T
                e = x[0:, :2]/x[0, 2] - x_2d[0:, 2*i:2*(i + 1)]
                self.assertLess(np.linalg.norm(e), 1)

if __name__ == "__main__":
    unittest.main()
