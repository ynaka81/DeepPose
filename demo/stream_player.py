import sys
from PyQt4 import QtGui

from stream_player.synchronized_stream import SynchronizedStream
from stream_player.stream_controller import StreamController
from stream_player.generate_progress import GenerateProgress
from stream_player.stream_generator import StreamGenerator
from stream_player.image_sequence_loader import ImageSequenceLoader

## Main windows
#
# The main windows of stream player demo
class MainWindow(QtGui.QMainWindow):
    ## constructor
    def __init__(self):
        super(MainWindow, self).__init__()
        ## synchronized stream
        self._synchronized_stream = None
        ## stream controller
        self._stream_controller = None
        ## generate progress
        self._generate_progress = None
        ## stream generator
        self._generator = None
        # setup
        self.initUI()
        self.initConnect()
    ## setup UI component
    # @param self The object pointer
    def initUI(self):
        # create main layout
        central_widget = QtGui.QWidget(self)
        main_layout = QtGui.QVBoxLayout(central_widget)
        self.setCentralWidget(central_widget)
        self._synchronized_stream = SynchronizedStream(central_widget)
        main_layout.addWidget(self._synchronized_stream)
        self._stream_controller = StreamController(central_widget)
        main_layout.addWidget(self._stream_controller)
        # create menu
        menubar = self.menuBar()
        file_menu = menubar.addMenu("&File")
        open_action = QtGui.QAction("&Open", self)
        open_action.setShortcut("Ctrl+O")
        open_action.setStatusTip("Open files")
        open_action.triggered.connect(self._openFiles)
        file_menu.addAction(open_action)
        exit_action = QtGui.QAction("&Exit", self)
        exit_action.setShortcut("Ctrl+Q")
        exit_action.setStatusTip("Exit application")
        exit_action.triggered.connect(QtGui.qApp.quit)
        file_menu.addAction(exit_action)
        edit_menu = menubar.addMenu("&Edit")
        set_action = QtGui.QAction("&Settings", self)
        set_action.setShortcut("Ctrl+E")
        set_action.setStatusTip("Set system parameter")
        set_action.triggered.connect(self._setParam)
        edit_menu.addAction(set_action)
        # create status bar
        statusbar = QtGui.QStatusBar(self)
        self._generate_progress = GenerateProgress(self)
        self._generate_progress.hide()
        statusbar.addWidget(self._generate_progress, 1)
        self.setStatusBar(statusbar)
        # create thread
        self._generator = StreamGenerator(self)
        # set window and show
        self.setWindowTitle("Stream player")
        self.show()
    ## connect signals and slots
    # @param self The object pointer
    def initConnect(self):
        # start stream generation
        self._generator.started.connect(self._generate_progress.show)
        # stream generating
        self._generator.updated.connect(self._synchronized_stream.updateStream)
        self._generator.updated.connect(self._stream_controller.updateStream)
        self._generator.updated.connect(self._generate_progress.updateStream)
        # finish stream generating
        self._generator.finished.connect(self._generate_progress.hide)
        # stream controller
        self._stream_controller.updated.connect(self._synchronized_stream.updateFrame)
    ## open files to browse
    # @param self The object pointer
    def _openFiles(self):
        filenames = [str(path) for path in QtGui.QFileDialog.getOpenFileNames(self, "Open files to browse")]
        # prepare loader for the stream type
        if len(filenames) > 1:
            loader = ImageSequenceLoader(filenames)
        # set sequence length to the other components
        frames = len(loader)
        self._stream_controller.setMaxFrames(frames)
        self._generate_progress.setMaxFrames(frames)
        # start generating 3D pose from the given image sequence
        self._generator.setup(loader)
        self._generator.start()
    ## set system parameter
    # @param self The object pointer
    def _setParam(self):
        pass
 
## start gui
def main():
    app = QtGui.QApplication(sys.argv)
    window = MainWindow()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()   
