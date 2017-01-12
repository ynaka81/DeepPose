from PyQt4 import QtGui

## Image stream widget
#
# The widget that draw image sequence
class ImageStream(QtGui.QLabel):
    ## constructor
    # @param parent The parent widget
    def __init__(self, parent=None):
        super(ImageStream, self).__init__(parent)
        ## image sequence
        self._image = []
        # setup UI
        self.initUI()
    ## setup UI component
    # @param self The object pointer
    def initUI(self):
        self.setStyleSheet("background-color : black;")
        self.setMinimumSize(227, 227)
    ## conver numpy array to QPixmap
    # @param self The object pointer
    # @param image The numpy array
    # @return QPixmap
    def _convertImage(self, image):
        h, w, _ = image.shape
        q_image = QtGui.QImage(image.data, w, h, w*3, QtGui.QImage.Format_RGB888)
        return QtGui.QPixmap(q_image)
    ## update image stream
    # @param self The object pointer
    # @param frame The current frame
    # @param image The image of the current frame
    def updateStream(self, frame, image):
        if frame == 0:
            self._image = [self._convertImage(image)]
            self.updateFrame(0)
        else:
            self._image.append(self._convertImage(image))
    ## update image frame slot
    # @param self The object pointer
    # @param frame The current frame
    def updateFrame(self, frame):
        self.setPixmap(self._image[frame])
