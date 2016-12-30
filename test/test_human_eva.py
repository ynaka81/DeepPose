import os
import random
import cv2

## Test HumanEva dataset
#
# The qualitative tester for HumanEva dataset
class TestHumanEva(object):
    # define const values
    N = 10
    ## constructor
    # @param train_file The filename of train file
    # @param result_dirname The directory for the test result
    # @param cropping_size The size of cropping for DNN training
    def __init__(self, train_file="data/train_data", result_dirname="test_result/HumanEva", cropping_size=227):
        for key, val in locals().items():
            setattr(self, key, val)
        try:
            os.makedirs(result_dirname)
        except OSError:
            pass
    ## main method of testing the HumanEva dataset
    # @param self The object pointer
    def main(self):
        # load train data
        lines = []
        for line in open(self.train_file):
            lines.append(line[:-1])
        lines = lines[:-1]
        # random test
        for i in range(self.N):
            line = random.choice(lines)
            splited = line.split(",")
            image_filename = splited[0]
            x_2d = zip(splited[16::5], splited[17::5])
            # draw image
            image = cv2.imread(image_filename)
            for j, x in enumerate(x_2d):
                x = [int(float(v)) for v in x]
                x_ = (x[0], x[1])
                cv2.circle(image, x_, 3, (0, 0, 255), -1)
                cv2.putText(image, str(j), x_, cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255))
            # cropping test
            h, w, _ = image.shape
            w_, h_ = w - self.cropping_size, h - self.cropping_size
            for j, u_0 in enumerate(((0, 0), (w_, 0), (0, h_), (w_, h_))):
                cv2.imwrite(os.path.join(self.result_dirname, "{0}_{1}.png".format(i, j)), image[u_0[0]:u_0[0] + self.cropping_size, u_0[1]:u_0[1] + self.cropping_size])

if __name__ == "__main__":
    TestHumanEva().main()
