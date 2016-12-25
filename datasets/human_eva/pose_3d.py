import numpy as np

import pose

## 3D pose
#
# The 3D pose computed from markers
class Pose3D(pose.Pose):
    ## constructor
    # @param Tm The global transform of markers
    # @param length The limb length of the current frame
    def __init__(self, Tm, length):
        super(Pose3D, self).__init__()
        self._data["torsoProximal"] = Tm.pelvis[0:3]*np.matrix([0, 0, length[0], 1]).T
        self._data["torsoDistal"] = Tm.pelvis[0:3, 3]
        self._data["upperLLegProximal"] = Tm.lthigh[0:3, 3]
        self._data["upperLLegDistal"] = Tm.lthigh[0:3]*np.matrix([0, 0, length[2], 1]).T
        self._data["lowerLLegProximal"] = Tm.ltibia[0:3, 3]
        self._data["lowerLLegDistal"] = Tm.ltibia[0:3]*np.matrix([0, 0, length[3], 1]).T
        self._data["upperRLegProximal"] = Tm.rthigh[0:3, 3]
        self._data["upperRLegDistal"] = Tm.rthigh[0:3] *np.matrix([0, 0, length[6], 1]).T
        self._data["lowerRLegProximal"] = Tm.rtibia[0:3, 3]
        self._data["lowerRLegDistal"] = Tm.rtibia[0:3]*np.matrix([0, 0, length[7], 1]).T
        self._data["upperLArmProximal"] = Tm.lshoulder[0:3, 3]
        self._data["upperLArmDistal"] = Tm.lshoulder[0:3]*np.matrix([0, 0, -length[10], 1]).T
        self._data["lowerLArmProximal"] = Tm.lelbow[0:3, 3]
        self._data["lowerLArmDistal"] = Tm.lelbow[0:3]*np.matrix([0, 0, -length[11], 1]).T
        self._data["upperRArmProximal"] = Tm.rshoulder[0:3, 3]
        self._data["upperRArmDistal"] = Tm.rshoulder[0:3]*np.matrix([0, 0, -length[14], 1]).T
        self._data["lowerRArmProximal"] = Tm.relbow[0:3, 3]
        self._data["lowerRArmDistal"] = Tm.relbow[0:3]*np.matrix([0, 0, -length[15], 1]).T
        self._data["headProximal"] = Tm.head[0:3, 3]
        self._data["headDistal"] = Tm.head[0:3]*np.matrix([0, 0, length[17], 1]).T
