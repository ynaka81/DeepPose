# DeepPose

This is not official implementation. Original paper is [DeepPose: Human Pose Estimation via Deep Neural Networks](http://arxiv.org/abs/1312.4659).
And the implementation model performs 3D pose esimation.

# Requirements

- Python 2.7.6
- [Chainer 1.19.0](https://github.com/pfnet/chainer)
- NumPy 1.11.3
- six 1.10.0
- Pillow 3.4.1
- SciPy 0.17.1
- OpenCV 2.4.8
- pyquaternion 0.9.0
- paramiko 1.16.0
- scp 0.10.2
- matplotlib 1.5.1
- PyQt (QtCore, QtGui, QtOpenGL) 4.10.4 
- PyOpenGL 3.1.0

# Usage

## Dataset preparation

First, download [HumanEva Dataset](http://humaneva.is.tue.mpg.de) to 'DeepPose/orig_data'.
And execute the folowing script.

```
python datasets/human_eva.py
python datasets/compute_mean.py
```

`human_eva.py` performs to calculate bounding boxes of human, resize images, and modify 2D/3D poses for your Neural Networks training.
`compute_mean.py` performs to compute a mean image of the datasets.

## Start training

Just run:

```
python train/train_pose_net.py
```

If you want to run `train_pose_net.py` with your own settings, please check the options first by `python train/train_pose_net.py --help` and give customize training settings.

## Visualize log

Just run:

```
python visualize/visualize_log.py
```

If you run `train_pose_net.py` on some clouds, please specify the ip address with `python visialize/visualize_log.py --locate 'username@xxx.xxx.xxx.xxx'`.
And you can see the other settings with `python visialize/visualize_log.py --help`.

## Prediction

Execute the following scripts, GUI tool will start.

```
python demo/stream_player.py
```
