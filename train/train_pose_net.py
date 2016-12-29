import os
import argparse
import imp
import numpy as np
from chainer import iterators
from chainer import optimizers
from chainer import serializers
from chainer import training
from chainer.training import extensions
import chainer

from pose_dataset import PoseDataset

## The supplement class
#
# The supplement class for validating the net
class TestModeEvaluator(extensions.Evaluator):
    ## validate the net
    # @param args The command line arguments
    def evaluate(self):
        model = self.get_target("main")
        model.train = False
        ret = super(TestModeEvaluator, self).evaluate()
        model.train = True
        return ret

## Train 2D/3D pose net
#
# The training the neural network fof estimating 2D/3D pose
class TrainPoseNet(object):
    ## constructor
    # @param args The command line arguments
    def __init__(self, args):
        ## command line arguments
        self.__args = args
    ## main method of generating the HumanEva dataset
    # @param self The object pointer
    def main(self):
        args = self.__args
        # initialize model to train
        model_name  = os.path.basename(args.model).split(".")[0]
        model_class = "".join([s.capitalize() for s in model_name.split("_")])
        model = imp.load_source(model_name, args.model)
        model = getattr(model, model_class)
        model = model(args.Nj)
        if args.resume_model:
            chainer.serializers.load_npz(args.resume_model, model)
        # prepare gpu
        if args.gpu >= 0:
            chainer.cuda.get_device(args.gpu).use()
            model.to_gpu()
        # load the datasets and mean file
        mean = np.load(args.mean)
        train = PoseDataset(args.train, mean)
        val = PoseDataset(args.val, mean, data_augmentation=False)
        # training/validation iterators
        train_iter = chainer.iterators.MultiprocessIterator(train, args.batchsize)
        val_iter = chainer.iterators.MultiprocessIterator(val, args.batchsize, repeat=False, shuffle=False)
        # Set up an optimizer
        optimizer = chainer.optimizers.MomentumSGD(lr=0.01, momentum=0.9) #TODO: tune
        optimizer.setup(model)
        if args.resume_opt:
            chainer.serializers.load_npz(args.resume_opt, optimizer)
        # Set up a trainer
        updater = training.StandardUpdater(train_iter, optimizer, device=args.gpu)
        trainer = training.Trainer(updater, (args.epoch, "epoch"), os.path.join(args.out, model_class))
        # standard trainer settings
        trainer.extend(extensions.dump_graph("main/loss"))
        val_interval = (1 if args.test else 10), "epoch"
        trainer.extend(TestModeEvaluator(val_iter, model, device=args.gpu), trigger=val_interval)
        # save parameters and optimization state per validation step
        trainer.extend(extensions.snapshot_object(model, "epoch-{.updater.epoch}.model"), trigger=val_interval)
        trainer.extend(extensions.snapshot_object(optimizer, "epoch-{.updater.epoch}.state"), trigger=val_interval)
        trainer.extend(extensions.snapshot(filename="epoch-{.updater.epoch}.iter"), trigger=val_interval)
        # show log
        log_interval = (1 if args.test else 10), "iteration"
        trainer.extend(extensions.LogReport(trigger=log_interval))
        trainer.extend(extensions.observe_lr(), trigger=log_interval)
        trainer.extend(extensions.PrintReport(["epoch", "main/loss", "validation/main/loss", "lr"]), trigger=log_interval)
        trainer.extend(extensions.ProgressBar(update_interval=10))
        # start training
        if args.resume:
            chainer.serializers.load_npz(args.resume, trainer)
        trainer.run()
 
if __name__ == "__main__":
    # arg definition
    parser = argparse.ArgumentParser(description="Learning convnet for estimating 2D/3D pose")
    parser.add_argument("--model", "-m", type=str, default="models/stub_net.py", help="Model definition file in models dir")
    parser.add_argument("--gpu", "-g", type=int, default=-1, help="GPU ID (negative value indicates CPU")
    parser.add_argument("--epoch", "-e", type=int, default=100, help="Number of epochs to train")
    parser.add_argument("--test", action="store_true", default=False, help="True when you would test something")
    parser.add_argument("--train", type=str, default="data/train_data", help="Path to training image-pose list file")
    parser.add_argument("--val", type=str, default="data/test_data", help="Path to validation image-pose list file")
    parser.add_argument("--mean", type=str, default="data/mean.npy", help="Mean image file (computed by compute_mean.py)")
    parser.add_argument('--Nj', type=int, default=20, help="Number of joints")
    parser.add_argument("--batchsize", type=int, default=32, help="Learning minibatch size")
    parser.add_argument("--out", default="result", help="Output directory")
    parser.add_argument("--resume", default=None, help="Initialize the trainer from given file. The file name is 'epoch-{.updater.epoch}.iter'")
    parser.add_argument("--resume_model", type=str, default=None, help="Load model definition file to use for resuming training (it\'s necessary when you resume a training). The file name is 'epoch-{.updater.epoch}.model'")
    parser.add_argument("--resume_opt", type=str, default=None, help="Load optimization states from this file (it\'s necessary when you resume a training). The file name is 'epoch-{.updater.epoch}.state'")
    TrainPoseNet(parser.parse_args()).main()
