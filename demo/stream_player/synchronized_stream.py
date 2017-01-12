from PyQt4 import QtGui

from stream_player.image_stream import ImageStream
from stream_player.pose_stream import PoseStream

## Synchronized stream widget
#
# The widget that synchronizes ImageStream and PoseStream
class SynchronizedStream(QtGui.QWidget):
    ## constructor
    # @param parent The parent widget
    def __init__(self, parent=None):
        super(SynchronizedStream, self).__init__(parent)
        ## image stream
        self._image_stream = None
        ## pose stream
        self._pose_stream = None
        # setup UI
        self.initUI()
    ## setup UI component
    # @param self The object pointer
    def initUI(self):
        # create main layout
        layout = QtGui.QHBoxLayout(self)
        self.setLayout(layout)
        self._image_stream = ImageStream(self)
        layout.addWidget(self._image_stream)
        self._pose_stream = PoseStream(self)
        layout.addWidget(self._pose_stream)
    ## update stream slot
    # @param self The object pointer
    # @param stream The stream structor
    def updateStream(self, stream):
        self._image_stream.updateStream(stream.frame, stream.image)
        self._pose_stream.updateStream(stream.frame, stream.pose)
    ## update synchronized stream frame slot
    # @param self The object pointer
    # @param frame The current frame
    def updateFrame(self, frame):
        self._image_stream.updateFrame(frame)
        self._pose_stream.updateFrame(frame)
