import os
import sys
import numpy as np
from PIL import Image

## Compute mean image
#
# The class for computing mean image
class ComputeMeanImage(object):
    ## constructor
    # @param train_file The filename of train file, the default value is "train_data"
    # @param output_file The filename of output file, the default value is "mean.npy"
    def __init__(self, train_file="data/train_data", output_file="data/mean.npy"):
        try:
            os.makedirs(os.path.dirname(output_file))
        except OSError:
            pass
        for k, v in locals().items():
            setattr(self, k, v)
    ## read image as numpy array
    # @param self The object pointer
    # @param path The path to image file
    # @return image array
    def __readImage(self, path):
        f = Image.open(path)
        try:
            image = np.asarray(f, dtype=np.float32)
        finally:
            f.close()
        return image.transpose(2, 0, 1)
    ## main method of generating the HumanEva dataset
    # @param self The object pointer
    def main(self):
        # read train file
        paths = []
        for line in open(self.train_file):
            paths.append(line.split(",")[0])
        paths = paths[:-1]
        # compute mean image
        N = len(paths)
        sum_image = 0
        for i, path in enumerate(paths):
            image = self.__readImage(path)
            sum_image += image
            sys.stderr.write("{0} / {1}\r".format(i + 1, N))
            sys.stderr.flush()
        sys.stderr.write("\n")
        mean_image = sum_image/N
        # save mean image
        np.save(self.output_file, mean_image)

if __name__ == "__main__":
    ComputeMeanImage().main()
