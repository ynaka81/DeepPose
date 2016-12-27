import numpy as np
import matplotlib.pyplot as plt

## Test ComputeMeanImage
#
# The qualitative tester for ComputeMeanImage
class TestComputeMeanImage(object):
    ## constructor
    # @param mean_image_file The filename of mean image
    def __init__(self, mean_image_file="data/mean.npy"):
        self.__mean_image = np.load(mean_image_file).transpose(1, 2, 0)
    ## main method of testing the HumanEva dataset
    # @param self The object pointer
    def main(self):
        # display mean image
        plt.imshow(self.__mean_image/255., vmin=0., vmax=1.)
        plt.show()

if __name__ == "__main__":
    TestComputeMeanImage().main()
