import os
import numpy as np
import scipy.io
import cv2

import dataset
from human_eva.actor_parameter import ActorParameter
from human_eva.synchronization_parameter import SynchronizationParameter
from human_eva.motion_capture_data import MotionCaptureData
from human_eva.camera_parameter import CameraParameter
from human_eva.conic_limb_parameter import ConicLimbParameter
from human_eva.limb_length import LimbLength
from human_eva.global_marker_transform import GlobalMarkerTransform
from human_eva.pose_3d import Pose3D
from human_eva.pose_2d import Pose2D

# TODO:for test, remove later
import sys
# TODO:to here

## HumanEva dataset
#
# The loader class of HumanEva dataset : "L. Sigal, A. O. Balan and M. J. Black. HumanEva: Synchronized Video and Motion Capture Dataset for Evaluation of Articulated Human Motion, International Journal of Computer Vision (IJCV), Volume 87, Number 1-2, pp. 4-27, March, 2010."
class HumanEva(dataset.Dataset):
    # define const values
    ACTORS = ("S1", "S2", "S3")
    CAMERAS = ("C1", "C2", "C3")
    ACTIONS = ("Box", "Gestures", "Jog", "ThrowCatch", "Walking")
    TRIALS = (1,)
    PARTITION = {"S1":{"Walking":590, "Jog":367, "ThrowCatch": 473, "Gestures": 395, "Box": 385},
                 "S2":{"Walking":438, "Jog":398, "ThrowCatch": 550, "Gestures": 500, "Box": 382},
                 "S3":{"Walking":448, "Jog":401, "ThrowCatch": 493, "Gestures": 533, "Box": 512}}
    START_FRAME = 6
    # TODO:extract using joint
    MAPPING = {"torsoProximal": "torsoProximal",
               "torsoDistal": "torsoDistal",
               "upperLLegProximal": "upperLLegProximal",
               "upperLLegDistal": "upperLLegDistal",
               "lowerLLegProximal": "lowerLLegProximal",
               "lowerLLegDistal": "lowerLLegDistal",
               "upperRLegProximal": "upperRLegProximal",
               "upperRLegDistal": "upperRLegDistal",
               "lowerRLegProximal": "lowerRLegProximal",
               "lowerRLegDistal": "lowerRLegDistal",
               "upperLArmProximal": "upperLArmProximal",
               "upperLArmDistal": "upperLArmDistal",
               "lowerLArmProximal": "lowerLArmProximal",
               "lowerLArmDistal": "lowerLArmDistal",
               "upperRArmProximal": "upperRArmProximal",
               "upperRArmDistal": "upperRArmDistal",
               "lowerRArmProximal": "lowerRArmProximal",
               "lowerRArmDistal": "lowerRArmDistal",
               "headProximal": "headProximal",
               "headDistal": "headDistal"}
    ## constructor
    # @param input_dirname The directory name of the input HumanEva dataset root
    # @param output_dirname The directory name of the output image root
    def __init__(self, input_dirname="orig_data/HumanEva", output_dirname="data/images/HumanEva"):
        super(HumanEva, self).__init__()
        self.__input_dirname = input_dirname
        self.__output_dirname = output_dirname
    ## generate the dataset data
    # @param self The object pointer
    # @param avi_file The filename of video data
    # @param png_dir The directory name of output image data
    # @param partition The partition between train data and test data
    # @param cam_param The camera parameter
    # @param mocap The motion capture data
    # @param sync_param The synchronization parameter
    # @param conic_param The conic limb parameter
    def __generateData(self, avi_file, png_dir, partition, cam_param, mocap, sync_param, conic_param):
        try:
            os.makedirs(png_dir)
        except OSError:
            pass
        image_frame = self.START_FRAME - 1
        # load video
        video = cv2.VideoCapture(avi_file)
        video.set(cv2.cv.CV_CAP_PROP_POS_FRAMES, image_frame)
        while(video.isOpened()):
            # get current image frame
            ret, frame = video.read()
            # calculate mocap frame
            mocap_frame = sync_param.mc_st + (image_frame - sync_param.im_st)*sync_param.mc_sc
            if (not 1 <= mocap_frame <= mocap.marker.shape[0]) or not mocap.isValid(mocap_frame):
                image_frame += 1
                continue
            # save image
            filename = os.path.join(png_dir, "{0}.png".format(image_frame))
            cv2.imwrite(filename, frame)
            # compute 3D pose
            length = LimbLength(mocap, mocap_frame)
            Tm = GlobalMarkerTransform(mocap, mocap_frame, conic_param, length)
            x_3d_t = Pose3D(Tm, length)
            x_2d_t = Pose2D(x_3d_t, cam_param)
            # substitute to class value
            index = (0 if image_frame < partition else 1)
            self._images[index].append(filename)
            self._P_[index].append(np.matrix(np.zeros((3, 2)))) # TODO:implement P+
            self._x_2d[index].append(x_2d_t)
            self._x_3d[index].append(x_3d_t)
            # increment image frame
            image_frame += 1
            # TODO:for test, remove later
            if image_frame > 100:
                break
            # TODO:to here
        # release memory
        video.release()
    ## main method of generating the HumanEva dataset
    # @param self The object pointer
    def main(self):
        # crawl HumanEva dataset
        for actor in self.ACTORS:
            actor_param = ActorParameter(os.path.join(self.__input_dirname, actor, "Mocap_Data", "{0}.mp".format(actor)))
            for camera in self.CAMERAS:
                cam_param = CameraParameter(os.path.join(self.__input_dirname, actor, "Calibration_Data", camera + ".cal"))
                for trial in self.TRIALS:
                    for action in self.ACTIONS:
                        mocap = MotionCaptureData(os.path.join(self.__input_dirname, actor, "Mocap_Data", "{0}_{1}.mat".format(action, trial)))
                        sync_param = SynchronizationParameter(os.path.join(self.__input_dirname, actor, "Sync_Data", "{0}_{1}_({2}).ofs".format(action, trial, camera)))
                        conic_param = ConicLimbParameter(mocap, actor_param)
                        self.__generateData(os.path.join(self.__input_dirname, actor, "Image_Data", "{0}_{1}_({2}).avi".format(action, trial, camera)), os.path.join(self.__output_dirname, actor, action), self.PARTITION[actor][action], cam_param, mocap, sync_param, conic_param)
                        # TODO:for test, remove later
                        self._saveDataset()
                        sys.exit()
                        # TODO:to here
        self._saveDataset()


if __name__ == "__main__":
    HumanEva().main()
