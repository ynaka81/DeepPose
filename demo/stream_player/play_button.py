from PyQt4 import QtCore, QtGui

## Play/Stop button widget
#
# The push button that plays/stops the stream
class PlayButton(QtGui.QPushButton):
    # define const
    PLAY = 0
    STOP = 1
    # define signal
    play = QtCore.pyqtSignal()
    stop = QtCore.pyqtSignal()
    ## constructor
    # @param parent The parent widget
    def __init__(self, parent=None):
        super(PlayButton, self).__init__(parent)
        ## inner button state
        self._state = self.STOP
        # setup
        self.initUI()
        self.initConnect()
    ## setup UI component
    # @param self The object pointer
    def initUI(self):
        play_icon = QtGui.QApplication.style().standardIcon(QtGui.QStyle.SP_MediaPlay)
        self.setIcon(play_icon)
    ## connect signals and slots
    # @param self The object pointer
    def initConnect(self):
        self.clicked.connect(lambda x: self.changeState(1 - self._state))
    ## change state slot
    # @param self The object point
    # @param state The state to change
    # @return The previoud state
    def changeState(self, state):
        if state != self._state:
            if state == self.PLAY:
                icon = QtGui.QApplication.style().standardIcon(QtGui.QStyle.SP_MediaStop)
                self.play.emit()
            else:
                icon = QtGui.QApplication.style().standardIcon(QtGui.QStyle.SP_MediaPlay)
                self.stop.emit()
            self.setIcon(icon)
            self._state = state
            return 1 - state
        else:
            return state
