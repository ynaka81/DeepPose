from PyQt4 import QtCore, QtGui

## Seek bar widget
#
# The slider bar that seeks the stream
class SeekBar(QtGui.QSlider):
    ## constructor
    # @param parent The parent widget
    def __init__(self, parent=None):
        super(SeekBar, self).__init__(QtCore.Qt.Horizontal, parent)
        ## loaded frames
        self._loaded_frames = 0
        # setup UI
        self.initUI()
    ## setup UI component
    # @param self The object pointer
    def initUI(self):
        pass
    ## connect signals and slots
    # @param self The object pointer
    def initConnect(self):
        self.valueChanged.connect(self.modifyValue)
    ## modify seek position into loaded frames
    # @param self The object pointer
    # @param value The seek position
    def modifyValue(self, value):
        self.setValue(min(value, self._loaded_frames))
    ## update stream slot
    # @param self The object pointer
    # @param frames The loaded frames
    def updateStream(self, frames):
        self._loaded_frames = frames
