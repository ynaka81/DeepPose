import sys
import numpy as np
from PyQt4 import QtCore

from stream_player.stream import Stream

sys.path.append("./")
from estimators.deep_3d_pose import Deep3dPose

## Stream generator thread
#
# The 3D pose generator thread from image sequence
class StreamGenerator(QtCore.QThread):
    # define signal
    updated = QtCore.pyqtSignal(Stream)
    ## constructor
    # @param parent The parent widget
    def __init__(self, parent=None):
        super(StreamGenerator, self).__init__(parent)
        ## stream loader
        self._loader = None
        ## pose estimator
        self._estimator = Deep3dPose("models/alex_net.py", 14)
        self._estimator.init("local_model/AlexNet/epoch-10.model")
        ## the flag that controls the thread
        self._stopped = False
        ## mutex lock object
        self._mutex = QtCore.QMutex()
    ## setup the thread
    # @param parent The parent widget
    # @param loader The stream loader
    def setup(self, loader):
        self._loader = loader
        self._stopped = False
    ## stop the thread
    # @param parent The parent widget
    def stop(self):
        with QtCore.QMutexLocker(self._mutex):
            self._stopped = True
    ## the starting point for the thread
    # @param parent The parent widget
    def run(self):
        for frame, image in enumerate(self._loader):
            if self._stopped:
                return
            input_image = (image.transpose(2, 0, 1)/255).astype(np.float32)
            pose = self._estimator(input_image)
            self.updated.emit(Stream(frame, image, pose))
        self.stop()
        self.finished.emit()
