from PyQt4 import QtCore, QtGui

from stream_player.play_button import PlayButton
from stream_player.seek_bar import SeekBar
from stream_player.frame_label import FrameLabel

## Stream controller widget
#
# The widget that controls a SynchronizedStream
class StreamController(QtGui.QWidget):
    # define signal
    updated = QtCore.pyqtSignal(int)
    ## constructor
    # @param parent The parent widget
    def __init__(self, parent=None):
        super(StreamController, self).__init__(parent)
        ## play/stop button
        self._button = None
        ## previous button state
        self._previous = None
        ## seek bar
        self._seek_bar = None
        ## frame label
        self._label = None
        ## timer for stream control
        self._timer = None
        ## loaded frames
        self._loaded_frames = 0
        ## current frame
        self._frame = 0
        # setup
        self.initUI()
        self.initConnect()
    ## setup UI component
    # @param self The object pointer
    def initUI(self):
        # create main layout
        layout = QtGui.QHBoxLayout(self)
        self.setLayout(layout)
        self._button = PlayButton(self)
        layout.addWidget(self._button)
        self._seek_bar = SeekBar(self)
        layout.addWidget(self._seek_bar)
        self._label = FrameLabel(self)
        layout.addWidget(self._label)
        # create timer
        self._timer = QtCore.QTimer(self)
        self._timer.setInterval(1.e3/10)
    ## connect signals and slots
    # @param self The object pointer
    def initConnect(self):
        # play button
        self._button.play.connect(self._timer.start)
        self._button.stop.connect(self._timer.stop)
        # seek bar
        self._seek_bar.sliderPressed.connect(self._halt)
        self._seek_bar.valueChanged.connect(self.updateFrameBySeekBar)
        self._seek_bar.sliderReleased.connect(self._resume)
        # timer control
        self._timer.timeout.connect(self.updateFrameByTimer)
    ## halt the timer to seek
    # @param self The object pointer
    def _halt(self):
        self._previous = self._button.changeState(PlayButton.STOP)
    ## resume the timer
    # @param self The object pointer
    def _resume(self):
        self._button.changeState(self._previous)
    ## set maximum number of frames
    # @param self The object pointer
    # @param frames The maximum number of frames
    def setMaxFrames(self, frames):
        self._seek_bar.setRange(0, frames - 1)
        self._label.setMaxFrames(frames - 1)
    ## update stream slot
    # @param self The object pointer
    # @param stream The stream structor
    def updateStream(self, stream):
        self._loaded_frames = stream.frame
        self._seek_bar.updateStream(stream.frame)
    ## called when update stream frame by seek bar
    # @param self The object pointer
    # @param frame The current frame
    def updateFrameBySeekBar(self, frame):
        self._frame = frame
        self._label.updateFrame(frame)
        self.updated.emit(frame)
    ## called when update stream frame by timer
    # @param self The object pointer
    def updateFrameByTimer(self):
        if self._frame < self._loaded_frames:
            self._frame += 1
            self._seek_bar.setValue(self._frame)
            self._label.updateFrame(self._frame)
            self.updated.emit(self._frame)
        else:
            self._timer.stop()
            self._button.changeState(PlayButton.STOP)
