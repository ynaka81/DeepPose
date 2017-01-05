import argparse
import os
import sys
import glob
import imp
import re
import datetime
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

sys.path.append("./")
from train.pose_dataset import PoseDataset
from model_file_getter import ModelFileGetter

## Visualizing metrics for algorithm evaluation
#
# The metrics visualization tool for algorithm evaluation
class VisualizeMetrics(object):
    # define const values
    JOINTS = ("head", "neck", "thorax", "pelvis", "l_shoulder", "l_elbow", "l_wrist", "r_shoulder", "r_elbow", "r_wrist", "l_knee", "l_ankle", "r_knee", "r_ankle")
    ## constructor
    # @param args The command line arguments
    def __init__(self, args):
        ## command line arguments
        self.__args = args
    ## main method of generating the HumanEva dataset
    # @param self The object pointer
    def main(self):
        args = self.__args
        # argument validation
        target = args.target.split(",")
        estimator = args.estimator.split(",")
        locate = args.locate.split(",")
        epoch = args.epoch.split(",")
        if not(len(target) == len(estimator) == len(locate) == len(epoch)):
            raise AttributeError("The length of attributes is not the same, --target({0}), --estimator({1}), --locate({2}), --epoch({3}).".format(len(target), len(estimator), len(locate), len(epoch)))
        # if target is one file, start with evaluating an algorithm mode(0). if not, start with comparing algorithms mode(1).
        mode = 0 if len(target) == 1 else 1
        # load the datasets and mean file
        mean = np.load(args.mean)
        dataset = [PoseDataset(args.eval, mean, data_augmentation=False)]
        if mode == 0:
            dataset.append(PoseDataset(args.test, mean, data_augmentation=False))
        # load estimators
        estimators = []
        for estimator_i, target_i in zip(estimator, target):
            module_name  = os.path.basename(estimator_i).split(".")[0]
            class_name = "".join([s.capitalize() for s in module_name.split("_")])
            _estimator = imp.load_source(module_name, estimator_i)
            _estimator = getattr(_estimator, class_name)
            estimators.append(_estimator(target_i, args.Nj))
        # get model files
        model_class_name = [c.model.__class__.__name__ for c in estimators]
        model_file_dir = ModelFileGetter().get(locate, model_class_name)
        # loop with models
        pattern = re.compile("epoch-(\d+)\.model")
        E_all_epoch = []
        E_epoch_i = []
        case = []
        for i, (epoch_i, estimator_i, model_file_dir_i) in enumerate(zip(epoch, estimators, model_file_dir)):
            case.append([[], []])
            # loop with dataset
            for j, dataset_j in enumerate(dataset):
                E_all_epoch.append([[], []])
                # loop with epoch
                model_file_list = glob.glob(os.path.join(model_file_dir_i, "epoch-*.model"))
                model_file_list = sorted(model_file_list, cmp=lambda s1, s2: cmp(*(int(pattern.search(x).group(1)) for x in (s1, s2))))
                if epoch_i == "":
                    epoch_i_model = model_file_list[-1]
                else:
                    epoch_i_model = "epoch-{0}.model".format(epoch_i)
                for k, model_file_k in enumerate(model_file_list):
                    # init model with epoch-*.model
                    estimator_i.init(model_file_k)
                    E = np.zeros((len(dataset_j), args.Nj))
                    for l in range(len(dataset_j)):
                        image, A, x_2d, x_3d = dataset_j.get_example(l)
                        x_3d = np.matrix(x_3d)
                        y = estimator_i(image)
                        for m, y_m in enumerate(y):
                            # TODO:calculate relative position error
                            e = (y_m - x_3d[:, 3*m: 3*(m + 1)].T)*1000.
                            E[l, m] = np.linalg.norm(e)
                        # logging
                        sys.stderr.write("calculating... estimators({0}/{1}) dataset({2}/{3}) epoch({4}/{5}) data({6}/{7})\r".format(i, len(estimators), j, len(dataset), k, len(model_file_list), l, len(dataset_j)))
                        sys.stderr.flush()
                    # keep the epoch
                    E_all_epoch[len(dataset)*i + j][0].append(int(pattern.search(model_file_k).group(1)))
                    E_all_epoch[len(dataset)*i + j][1].append(E.mean())
                    if model_file_k == epoch_i_model:
                        E_epoch_i.append(E)
                        # calculate best/worst estimation and corresponding result
                        if j == 0:
                            E_mean = E.mean(axis=1)
                            l_min_max = [np.argpartition(E_mean, 3)[:3], np.argpartition(E_mean, -3)[-3:]]
                            for l in range(2):
                                for index in l_min_max[l]:
                                    image, A, x_2d, x_3d = dataset_j.get_example(index)
                                    A_m = np.matrix(A)
                                    y_3d = estimator_i(image)
                                    y_2d = []
                                    for y_3d_j in y_3d:
                                        p = A*y_3d_j
                                        y_2d.append(p[0:2]/p[2, 0])
                                    case[i][l].append((image, np.array(np.vstack(y_2d))[:,0]))
        sys.stderr.write("\n")
        # draw graphs
        rows = 1 if mode == 0 else len(E_all_epoch)
        r = 1.0/(2*rows + 6.3)
        f = lambda x: 1 - x*r
        gs1 = gridspec.GridSpec(1, 2, top=f(0.3), bottom=f(2.3))
        gs2 = gridspec.GridSpec(1, 2, top=f(3.2), bottom=f(5.2))
        gs3 = gridspec.GridSpec(rows, 2, top=f(6.2), bottom=f(2*rows + 6.2))
        fig = plt.figure(figsize=(15, 2*rows + 6.3))
        legend = ["{0}.{1}".format(c.__class__.__name__, c.model.__class__.__name__) for c in estimators]
        if mode == 0:
            legend.append("test data")
        color = lambda i: ("b", "g", "r", "c", "m", "y")[i%6]
        # draw loss function
        loss_plot = plt.subplot(gs1[0, 0])
        for i, e in enumerate(E_all_epoch):
            loss_plot.plot(e[0], e[1], color=color(i))
        loss_plot.set_yscale("log")
        loss_plot.legend(legend)
        loss_plot.set_title("MPJPE (mean per joint position error) per epoch")
        loss_plot.set_xlabel("epoch")
        loss_plot.set_ylabel("MPJPE [mm]")
        # time series
        time_series_plot = plt.subplot(gs1[0, 1])
        for i, e in enumerate(E_epoch_i):
            time_series_plot.plot(range(e.shape[0]), e.mean(axis=1), color=color(i))
        time_series_plot.legend(legend)
        time_series_plot.set_title("MPJPE (mean per joint position error) per frame")
        time_series_plot.set_xlabel("frame")
        time_series_plot.set_ylabel("MPJPE [mm]")
        # absolute position error plot
        abs_plot = plt.subplot(gs2[0, 0])
        Ne = len(E_epoch_i)
        d = 1.8/(3*Ne - 1)
        m = (Ne + 1)/2
        for i, e in enumerate(E_epoch_i):
            abs_plot.bar([j + d*(i - m) for j in range(e.shape[1])], e.mean(axis=0), yerr=e.std(axis=0), align="center", width=0.7/Ne, color=color(i), ecolor="k")
        abs_plot.legend(legend)
        abs_plot.set_xticks(range(len(self.JOINTS)))
        abs_plot.set_xticklabels(self.JOINTS, rotation=90, fontsize="small")
        abs_plot.set_xlim([-1, len(self.JOINTS)])
        abs_plot.set_title("joint absolute position error")
        abs_plot.set_ylabel("error [mm]")
        # relative position error plot(TODO:implement)
        rel_plot = plt.subplot(gs2[0, 1])
        # draw qualitative image plot
        offset = 10
        for i, case_i in enumerate(case):
            # loop with case (0:best, 1:worst)
            for j, case_ij in enumerate(case_i):
                image_list = []
                x_2d_list = [[], []]
                for k, case_ijk in enumerate(case_ij):
                    image, x_2d = case_ijk
                    _, h, w = image.shape
                    _, h_m, w_m = mean.shape
                    top, left = (h_m - h)/2, (w_m - h)/2
                    bottom, right = top + h, left + w
                    image = (image + mean[:, top:bottom, left:right]/255).transpose(1, 2, 0)
                    image_list.append(image)
                    if k != len(case_ij) -1:
                        image_list.append(np.zeros((h, offset, 3)))
                    for l in range(2):
                        x_2d_list[l] += map(int, x_2d[l::2] + (w + offset)*k*(l == 0))
                image_plot = plt.subplot(gs3[i, j])
                image_plot.imshow(np.concatenate(image_list, axis=1), vmin=0., vmax=1.)
                image_plot.scatter(x_2d_list[0], x_2d_list[1], color="r", s=5)
                image_plot.axis("off")
                image_plot.set_title("{0}-3 estimated image ({1})".format(("best", "worst")[j], legend[i]))
        # save graphs
        try:
            os.makedirs(args.out)
        except OSError:
            pass
        date = datetime.datetime.today()
        plt.savefig(os.path.join(args.out, "{0}-{1}-{2}-{3}-{4}.png".format(date.strftime("%Y%m%d-%H:%M:%S"), args.target.replace("/", "."), args.estimator.replace("/", "."), args.locate, args.epoch)))
        # show graphs
        plt.show()
 
if __name__ == "__main__":
    # arg definition
    parser = argparse.ArgumentParser(description="Visualizing metrics for algorithm evaluation")
    parser.add_argument("--target", "-t", type=str, default="models/alex_net.py", help="Target models definition file for algorithm evaluation")
    parser.add_argument("--estimator", "-s", type=str, default="estimators/deep_3d_pose.py", help="Target estimators definition file for algorithm evaluation")
    parser.add_argument("--locate", "-l", type=str, default="local", help="Location of model definition file ('local':use local file, 'username@xxx.xxx.xxx.xxx':get remote file and use it)")
    parser.add_argument("--epoch", "-e", type=str, default="", help="Epoch used for evaluation, the default is the newest")
    parser.add_argument("--eval", type=str, default="data/test_data", help="Path to evaluating image-pose list file")
    parser.add_argument("--test", type=str, default="data/test_data", help="Path to testing image-pose list file")
    parser.add_argument("--mean", type=str, default="data/mean.npy", help="Mean image file (computed by compute_mean.py)")
    parser.add_argument('--Nj', type=int, default=14, help="Number of joints")
    parser.add_argument("--out", default="result/metrics", help="Output directory")
    VisualizeMetrics(parser.parse_args()).main()
