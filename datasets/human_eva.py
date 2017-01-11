import sys
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
from human_eva.bounding_box import BoundingBox

## HumanEva dataset
#
# The loader class of HumanEva dataset : "L. Sigal, A. O. Balan and M. J. Black. HumanEva: Synchronized Video and Motion Capture Dataset for Evaluation of Articulated Human Motion, International Journal of Computer Vision (IJCV), Volume 87, Number 1-2, pp. 4-27, March, 2010."
class HumanEva(dataset.Dataset):
    # define const values
    ACTORS = ("S1", "S2", "S3")
    CAMERAS = ("C1", "C2", "C3")
    ACTIONS = ("Box", "Gestures", "Jog", "ThrowCatch", "Walking")
    TRIALS = (1,)
    PARTITION = {"S1":{"Walking": (590, 1180), "Jog": (367, 735), "ThrowCatch": (473, 946), "Gestures": (395, 790), "Box": (385, 770)},
                 "S2":{"Walking": (438, 877), "Jog": (398, 796), "ThrowCatch": (550, 1101), "Gestures": (500, 1000), "Box": (382, 765)},
                 "S3":{"Walking": (448, 896), "Jog": (401, 803), "ThrowCatch": (493, 987), "Gestures": (533, 1067), "Box": (512, 1024)}}
    START_FRAME = 6
    MAPPING = {"head": "headDistal",
               "neck": "headProximal",
               "thorax": "torsoProximal",
               "pelvis": "torsoDistal",
               "l_shoulder": "upperLArmProximal",
               "l_wrist": "lowerLArmDistal",
               "r_shoulder": "upperRArmProximal",
               "r_wrist": "lowerRArmDistal",
               "l_ankle": "lowerLLegDistal",
               "r_ankle": "lowerRLegDistal"}
    ## constructor
    # @param input_dirname The directory name of the input HumanEva dataset root
    # @param output_dirname The directory name of the output image root
    def __init__(self, input_dirname="orig_data/HumanEva", output_dirname="data/images/HumanEva"):
        super(HumanEva, self).__init__()
        self.__input_dirname = input_dirname
        self.__output_dirname = output_dirname
    ## map HumanEva joint list to DeepPose joint list
    # @param self The object pointer
    # @param pose The 2D/3D poses
    # @return The dictionary of DeepPose style joint
    def __mapJoint(self, pose):
        x = {k: None for k in self.JOINTS}
        for name in self.JOINTS:
            # simple name mapping
            if name in self.MAPPING:
                x[name] = pose[self.MAPPING[name]]
            # calculate DeepPose joint pose
            else:
                if "elbow" in name:
                    if name[0] == "l":
                        x[name] = (pose["upperLArmDistal"] + pose["lowerLArmProximal"])/2
                    else:
                        x[name] = (pose["upperRArmDistal"] + pose["lowerRArmProximal"])/2
                elif "knee" in name:
                    if name[0] == "l":
                        x[name] = (pose["upperLLegDistal"] + pose["lowerLLegProximal"])/2
                    else:
                        x[name] = (pose["upperRLegDistal"] + pose["lowerRLegProximal"])/2
        return x
    ## generate the dataset data
    # @param self The object pointer
    # @param avi_file The filename of video data
    # @param png_dir The directory name of output image data
    # @param partition The partition between test data, train data and video end
    # @param cam_param The camera parameter
    # @param mocap The motion capture data
    # @param sync_param The synchronization parameter
    # @param conic_param The conic limb parameter
    # @param log The log
    def __generateData(self, avi_file, png_dir, partition, cam_param, mocap, sync_param, conic_param, log):
        try:
            os.makedirs(png_dir)
        except OSError:
            pass
        image_frame = self.START_FRAME - 1
        # load video
        video = cv2.VideoCapture(avi_file)
        video.set(cv2.cv.CV_CAP_PROP_POS_FRAMES, image_frame)
        while(video.isOpened()):
            if image_frame > partition[1]:
                break
            # logging
            sys.stderr.write("{0} frames({1}/{2})\r".format(log, image_frame, partition[1]))
            sys.stderr.flush()
            # get current image frame
            ret, frame = video.read()
            if not ret:
                image_frame += 1
                continue
            # calculate mocap frame
            mocap_frame = sync_param.mc_st + (image_frame - sync_param.im_st)*sync_param.mc_sc
            if (not 1 <= mocap_frame <= mocap.marker.shape[0]) or not mocap.isValid(mocap_frame):
                image_frame += 1
                continue
            # compute 3D pose
            length = LimbLength(mocap, mocap_frame)
            Tm = GlobalMarkerTransform(mocap, mocap_frame, conic_param, length)
            x_3d_t = Pose3D(Tm, length)
            x_2d_t = Pose2D(x_3d_t, cam_param)
            # calculate bounding box
            try:
                bb = BoundingBox(frame, x_2d_t)
            except RuntimeError:
                image_frame += 1
                continue
            # modify 2D/3D pose accoring to the bounding box
            x_3d_t.modify(bb, cam_param)
            x_2d_t.modify(bb)
            # save image
            filename = os.path.join(png_dir, "{0}.png".format(image_frame))
            cv2.imwrite(filename, bb.image)
            # substitute to class value
            index = (1 if image_frame < partition[0] else 0)
            self._images[index].append(filename)
            self._A[index].append(cam_param.A)
            self._x_2d[index].append(self.__mapJoint(x_2d_t))
            self._x_3d[index].append(self.__mapJoint(x_3d_t))
            # increment image frame
            image_frame += 1
        # release memory
        video.release()
    ## main method of generating the HumanEva dataset
    # @param self The object pointer
    def main(self):
        # crawl HumanEva dataset
        for i, actor in enumerate(self.ACTORS):
            actor_param = ActorParameter(os.path.join(self.__input_dirname, actor, "Mocap_Data", "{0}.mp".format(actor)))
            for j, camera in enumerate(self.CAMERAS):
                cam_param = CameraParameter(os.path.join(self.__input_dirname, actor, "Calibration_Data", camera + ".cal"))
                for k, trial in enumerate(self.TRIALS):
                    for l, action in enumerate(self.ACTIONS):
                        # logging
                        log = "generating... actors({0}/{1}) cameras({2}/{3}) trials({4}/{5}) actions({6}/{7})".format(i, len(self.ACTORS), j, len(self.CAMERAS), k, len(self.TRIALS), l, len(self.ACTIONS))
                        # if mocap data is valid, generate dataset
                        try:
                            mocap = MotionCaptureData(os.path.join(self.__input_dirname, actor, "Mocap_Data", "{0}_{1}.mat".format(action, trial)))
                        except RuntimeError:
                            continue
                        sync_param = SynchronizationParameter(os.path.join(self.__input_dirname, actor, "Sync_Data", "{0}_{1}_({2}).ofs".format(action, trial, camera)))
                        conic_param = ConicLimbParameter(mocap, actor_param)
                        self.__generateData(os.path.join(self.__input_dirname, actor, "Image_Data", "{0}_{1}_({2}).avi".format(action, trial, camera)), os.path.join(self.__output_dirname, actor, action, camera), self.PARTITION[actor][action], cam_param, mocap, sync_param, conic_param, log)
        sys.stderr.write("\n")
        self._saveDataset()

if __name__ == "__main__":
    HumanEva().main()
