from PyQt4 import QtGui

## Frame label widget
#
# The slider bar that seeks the stream
class FrameLabel(QtGui.QLabel):
    ## constructor
    # @param parent The parent widget
    def __init__(self, parent=None):
        super(FrameLabel, self).__init__(parent)
        ## max frames
        self._max_frames = 0
        # setup UI
        self.initUI()
    ## setup UI component
    # @param self The object pointer
    def initUI(self):
        self.setText("000 / 000 [frame]")
    ## set maximum number of frames
    # @param self The object pointer
    # @param frames The maximum number of frames
    def setMaxFrames(self, frames):
        self._max_frames = frames
        self.setText("000 / {0:03d} [frame]".format(frames))
    ## update frame text
    # @param self The object pointer
    # @param frame The current frame
    def updateFrame(self, frame):
        self.setText("{0:03d} / {1:03d} [frame]".format(frame, self._max_frames))
