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
    def __init__(self, train_file="data/train_data", result_dirname="test_result/HumanEva"):
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
            x_2d = zip(splited[7::5], splited[8::5])
            # draw image
            image = cv2.imread(image_filename)
            for j, x in enumerate(x_2d):
                x = [int(float(v)) for v in x]
                x_ = (x[0], x[1])
                cv2.circle(image, x_, 3, (0, 0, 255), -1)
                cv2.putText(image, str(j), x_, cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255))
            cv2.imwrite(os.path.join(self.result_dirname, "{0}.png".format(i)), image)

if __name__ == "__main__":
    TestHumanEva().main()
