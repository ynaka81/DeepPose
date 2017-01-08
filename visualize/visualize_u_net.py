import argparse
import os
import sys
import glob
import re
import paramiko
import scp
import shutil
import numpy as np
import matplotlib.pyplot as plt
import chainer

sys.path.append("./")
from train.pose_dataset import PoseDataset
from models.u_net import UNet
from models.functions.channel_wise_softmax_cross_entropy import ChannelWiseSoftmaxCrossEntropy

## Visualizing 2D pose estimated by U-Net
#
# The 2D pose visualization tool, which estimated by U-Net
class VisualizeUNet(object):
    # define const values
    JOINTS = ("head", "neck", "thorax", "pelvis", "l_shoulder", "l_elbow", "l_wrist", "r_shoulder", "r_elbow", "r_wrist", "l_knee", "l_ankle", "r_knee", "r_ankle")
    ## constructor
    # @param args The command line arguments
    def __init__(self, args):
        ## command line arguments
        self.__args = args
        self.ssh = paramiko.SSHClient()
        self.ssh.load_system_host_keys()
        try:
            os.makedirs(args.local_model)
        except OSError:
            pass
        try:
            os.makedirs(args.out)
        except OSError:
            pass
    ## get model file
    # @param self The object pointer
    # @param args The command line arguments
    # @return model file name
    def __getModelFile(self, args):
        # find model file
        if args.epoch == "":
            if args.locate in ("local", "localhost"):
                if args.locate == "local":
                    path = os.path.join(args.local_model, "epoch-*.model")
                else:
                    path = os.path.join(args.local, "epoch-*.model")
                model_file_list = glob.glob(path)
            else:
                user, host = args.locate.split("@")
                self.ssh.connect(host, username=user)
                _, out, _ = self.ssh.exec_command("ls {0}/epoch-*.model".format(args.remote))
                model_file_list = []
                for line in out:
                    model_file_list.append(os.path.basename(line[:-1]))
            if len(model_file_list) == 0:
                raise RuntimeError("No model files are generated.")
            pattern = re.compile("epoch-(\d+)\.model")
            model_file  = os.path.basename(sorted(model_file_list, cmp=lambda s1, s2: cmp(*(int(pattern.search(x).group(1)) for x in (s1, s2))))[-1])
        else:
            model_file = "epoch-{0}.model".format(args.epoch)
        # get model file
        if args.locate == "local":
            pass
        elif args.locate == "localhost":
            # remove old file and get new one
            if os.path.isfile(model_file):
                os.remove(os.path.join(args.local_model, model_file))
            shutil.copy(os.path.join(args.local, model_file), args.local_model)
        else:
            user, host = args.locate.split("@")
            self.ssh.connect(host, username=user)
            with scp.SCPClient(self.ssh.get_transport()) as scp_client:
                # remove old file and get new one
                if os.path.isfile(model_file):
                    os.remove(os.path.join(args.local_model, model_file))
                scp_client.get(os.path.join(args.remote, model_file), args.local_model)
        return os.path.join(args.local_model, model_file)
    # extract N-Best result
    # @param self The object pointer
    # @param N N-Best
    # @param model The trained U-Net model
    # @param dataset The dataset to evaluate
    # @return N-Best result index
    def __extractNBest(self, N, model, dataset):
        result = []
        for i, data in enumerate(dataset):
            sys.stderr.write("calculating... {0}/{1}\r".format(i, len(dataset)))
            sys.stderr.flush()
            loss = model(*(np.array([v]) for v in data))
            result.append(float(loss.data))
        sys.stderr.write("\n")
        return np.argpartition(result, N)[:N]
    ## main method of generating the HumanEva dataset
    # @param self The object pointer
    def main(self):
        args = self.__args
        # initialize model
        model = UNet(args.Nj)
        chainer.serializers.load_npz(self.__getModelFile(args), model)
        # load the datasets and mean file
        mean = np.load(args.mean)
        dataset = PoseDataset(args.eval, mean, data_augmentation=False)
        # extract N-Best result
        n_best = self.__extractNBest(args.n_best, model, dataset)
        # draw N-Best image
        F = ChannelWiseSoftmaxCrossEntropy()
        for i, index in enumerate(n_best):
            sys.stderr.write("drawing... {0}/{1}\r".format(i, args.n_best))
            sys.stderr.flush()
            # calculate distribution image
            image, A, x_2d, x_3d = dataset.get_example(index)
            dist = model.predict(np.array([image]))
            log_X, _, _ = F._computeSoftmax(dist.data)
            X = np.exp(log_X)
            dist_image = np.maximum(X.sum(axis=(0, 1))[0:-1, 0:-1], 0)
            # draw image
            _, h, w = mean.shape
            hd, wd = dist_image.shape
            top, left = (h - hd)/2, (w - wd)/2
            bottom, right = top + hd, left + wd
            plt.subplot(1, 2, 1)
            plt.imshow((image + mean[:, top:bottom, left:right]/255).transpose(1, 2, 0), vmin=0., vmax=1.)
            plt.title("original image")
            plt.axis("off")
            plt.subplot(1, 2, 2)
            plt.imshow(dist_image, vmin=0., vmax=1.)
            plt.title("joint location distribution map")
            plt.axis("off")
            plt.savefig(os.path.join(args.out, "{0}.png".format(i)))
        sys.stderr.write("\n")
 
if __name__ == "__main__":
    # arg definition
    parser = argparse.ArgumentParser(description="Visualizing metrics for algorithm evaluation")
    parser.add_argument("--n_best", "-n", type=int, default=5, help="Extract N-Best result")
    parser.add_argument("--locate", "-l", type=str, default="local", help="Location of model definition file ('local':use local file, 'username@xxx.xxx.xxx.xxx':get remote file and use it)")
    parser.add_argument("--epoch", "-e", type=str, default="", help="Epoch used for evaluation, the default is the newest")
    parser.add_argument("--eval", type=str, default="data/test_data", help="Path to evaluating image-pose list file")
    parser.add_argument("--mean", type=str, default="data/mean.npy", help="Mean image file (computed by compute_mean.py)")
    parser.add_argument("--Nj", type=int, default=14, help="Number of joints")
    parser.add_argument("--out", default="result/u_net", help="Output directory")
    parser.add_argument("--local_model", type=str, default="local_model/UNet", help="The directory name which is used for local model file")
    parser.add_argument("--local", type=str, default="result/UNet", help="The local directory name of training result")
    parser.add_argument("--remote", type=str, default="~/DeepPose/result/UNet", help="The remote directory name of training result")
    VisualizeUNet(parser.parse_args()).main()
