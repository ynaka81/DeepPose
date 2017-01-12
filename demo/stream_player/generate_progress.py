from PyQt4 import QtCore, QtGui

## Generate progress progress bar widget
#
# The progress bar that display the progress of image/pose stream generation
class GenerateProgress(QtGui.QWidget):
    ## constructor
    # @param parent The parent widget
    def __init__(self, parent=None):
        super(GenerateProgress, self).__init__(parent)
        ## progress bar
        self._progress = None
        # setup UI
        self.initUI()
    ## setup UI component
    # @param self The object pointer
    def initUI(self):
        # create main layout
        layout = QtGui.QHBoxLayout(self)
        self.setLayout(layout)
        layout.addStretch()
        layout.addWidget(QtGui.QLabel("generating..."), 0, QtCore.Qt.AlignRight)
        self._progress = QtGui.QProgressBar(self)
        layout.addWidget(self._progress, 0, QtCore.Qt.AlignRight)
    ## set maximum number of frames
    # @param self The object pointer
    # @param frames The maximum number of frames
    def setMaxFrames(self, frames):
        self._progress.setRange(0, frames)
    ## update stream slot
    # @param self The object pointer
    # @param stream The stream structor
    def updateStream(self, stream):
        self._progress.setValue(stream.frame + 1)
