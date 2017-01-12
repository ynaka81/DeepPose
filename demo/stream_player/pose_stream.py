import sys
from PyQt4 import QtOpenGL
from OpenGL import GL, GLU, GLUT

## Pose stream widget
#
# The widget that draw 3D pose
class PoseStream(QtOpenGL.QGLWidget):
    ## constructor
    # @param parent The parent widget
    def __init__(self, parent=None):
        super(PoseStream, self).__init__(parent)
        ## pose sequence
        self._pose = []
        ## current frame
        self._current = None
        # setup UI
        self.initUI()
    ## setup UI component
    # @param self The object pointer
    def initUI(self):
        self.setMinimumSize(227, 227)
    ## setup the OpenGL rendering context
    # @param self The object pointer
    def initializeGL(self):
        GLUT.glutInit(sys.argv)
        GL.glShadeModel(GL.GL_SMOOTH)
        GL.glDepthFunc(GL.GL_LESS)
        GL.glEnable(GL.GL_DEPTH_TEST)
        GL.glEnable(GL.GL_CULL_FACE)
    ## setup the OpenGL after the widget has been resized
    # @param self The object pointer
    # @param w The width of the widget
    # @param h The height of the widget
    def resizeGL(self, w, h):
        side = min(w, h)
        GL.glViewport((w - side) / 2, (h - side) / 2, side, side)
        GL.glMatrixMode(GL.GL_PROJECTION)
        GL.glLoadIdentity()
        GL.glFrustum(-0.5, +0.5, +0.5, -0.5, 0.1, 10.0)
        GL.glMatrixMode(GL.GL_MODELVIEW)
    ## render the OpenGL scene
    # @param self The object pointer
    def paintGL(self):
        # prepare OpenGL window
        GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)
        GL.glLoadIdentity()
        GL.glPolygonMode(GL.GL_FRONT, GL.GL_FILL)
        # draw 3D pose
        if self._current is not None:
            GL.glColor3f(1.0, 1.5, 0.0)
            for x in self._pose[self._current]:
                GL.glPushMatrix()
                GL.glTranslatef(x[0, 0], x[1, 0], -x[2, 0])
                GLUT.glutSolidSphere(0.1, 32, 32)
                GL.glPopMatrix()
    ## update 3D pose stream
    # @param self The object pointer
    # @param frame The current frame
    # @param pose The 3D pose of the current frame
    def updateStream(self, frame, pose):
        if frame == 0:
            self._pose = [pose]
            self.updateFrame(0)
        else:
            self._pose.append(pose)
    ## update image frame slot
    # @param self The object pointer
    # @param frame The current frame
    def updateFrame(self, frame):
        self._current = frame
        self.updateGL()
