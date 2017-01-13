import sys
import numpy as np
from PyQt4 import QtCore, QtOpenGL
from OpenGL import GL, GLU, GLUT

sys.path.append("./")
from datasets.dataset import Dataset

## Pose stream widget
#
# The widget that draw 3D pose
class PoseStream(QtOpenGL.QGLWidget):
    # define const values
    CHAIN_BODY = ((Dataset.JOINTS.index("neck"), Dataset.JOINTS.index("thorax")),
                  (Dataset.JOINTS.index("thorax"), Dataset.JOINTS.index("pelvis")),
                  (Dataset.JOINTS.index("thorax"), Dataset.JOINTS.index("l_shoulder")),
                  (Dataset.JOINTS.index("l_shoulder"), Dataset.JOINTS.index("l_elbow")),
                  (Dataset.JOINTS.index("l_elbow"), Dataset.JOINTS.index("l_wrist")),
                  (Dataset.JOINTS.index("thorax"), Dataset.JOINTS.index("r_shoulder")),
                  (Dataset.JOINTS.index("r_shoulder"), Dataset.JOINTS.index("r_elbow")),
                  (Dataset.JOINTS.index("r_elbow"), Dataset.JOINTS.index("r_wrist")),
                  (Dataset.JOINTS.index("pelvis"), Dataset.JOINTS.index("l_knee")),
                  (Dataset.JOINTS.index("l_knee"), Dataset.JOINTS.index("l_ankle")),
                  (Dataset.JOINTS.index("pelvis"), Dataset.JOINTS.index("r_knee")),
                  (Dataset.JOINTS.index("r_knee"), Dataset.JOINTS.index("r_ankle")))
    CHAIN_HEAD = (Dataset.JOINTS.index("head"), Dataset.JOINTS.index("neck"))
    HEAD, BODY = 0, 1
    ## constructor
    # @param parent The parent widget
    def __init__(self, parent=None):
        super(PoseStream, self).__init__(parent)
        ## pose sequence
        self._pose = []
        ## current frame
        self._current = None
        ## rotation vector
        self._rot = np.zeros(3)
        ## last mouse click position
        self._last = None
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
        GL.glEnable(GL.GL_LIGHTING)
        GL.glEnable(GL.GL_LIGHT0)
        GL.glLightfv(GL.GL_LIGHT0, GL.GL_POSITION, (1, -1, -1, 0))
    ## setup the OpenGL after the widget has been resized
    # @param self The object pointer
    # @param w The width of the widget
    # @param h The height of the widget
    def resizeGL(self, w, h):
        side = min(w, h)
        GL.glViewport((w - side)/2, (h - side)/2, side, side)
        GL.glMatrixMode(GL.GL_PROJECTION)
        GL.glLoadIdentity()
        GL.glFrustum(-0.45, -0.05, 0.05, -0.35, 1, 10)
        GL.glMatrixMode(GL.GL_MODELVIEW)
    ## draw sphere(head) or cylinder(body) between two points
    # @param self The object pointer
    # @param bone_type The type of bone
    # @param x1 The point1
    # @param x2 The point2
    def _drawBone(self, bone_type, x1, x2):
        d = x2 - x1
        n = np.linalg.norm(d)
        GL.glPushMatrix()
        GL.glTranslatef(x1[0, 0], x1[1, 0], -x1[2, 0])
        # ignore when the vector d is nearly +-e_z
        if np.abs(d[2, 0]/n) < 1 - 1.e-5:
            rad = np.arccos(-d[2, 0]/n)
            GL.glRotatef(np.rad2deg(rad), -d[1, 0], d[0, 0], 0)
        if bone_type == self.HEAD:
            GL.glTranslatef(0, 0, n/3)
            GLUT.glutSolidSphere(n/3, 32, 32)
        elif bone_type == self.BODY:
            GLUT.glutSolidCylinder(1.e-2, n, 32, 2)
        GL.glPopMatrix()
    ## set color to OpenGL
    # @param self The object pointer
    # @param color The color to draw (R,G,B)
    def _setColor(self, color):
        GL.glMaterialfv(GL.GL_FRONT_AND_BACK, GL.GL_AMBIENT, color)
        GL.glMaterialfv(GL.GL_FRONT_AND_BACK, GL.GL_DIFFUSE, color)
        GL.glMaterialfv(GL.GL_FRONT_AND_BACK, GL.GL_SPECULAR, color)
        GL.glMaterialfv(GL.GL_FRONT_AND_BACK, GL.GL_SHININESS, 10)
    ## render the OpenGL scene
    # @param self The object pointer
    def paintGL(self):
        # prepare OpenGL window
        GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)
        GL.glLoadIdentity()
        # camera translation
        GL.glRotated(self._rot[0], 1, 0, 0)
        GL.glRotated(self._rot[1], 0, 1, 0)
        GL.glRotated(self._rot[2], 0, 0, 1)
        # draw 3D pose
        if self._current is not None:
            pose = self._pose[self._current]
            # draw bone
            self._setColor((0, 1, 0))
            for l in self.CHAIN_BODY:
                self._drawBone(self.BODY, pose[l[0]], pose[l[1]])
            self._setColor((1, 1, 1))
            x_head = pose[self.CHAIN_HEAD[0]]
            x_neck = pose[self.CHAIN_HEAD[1]]
            self._drawBone(self.HEAD, x_head, x_neck)
            self._setColor((0, 1, 0))
            self._drawBone(self.BODY, x_neck, x_neck + (x_head - x_neck)/3)
            # draw joint
            self._setColor((1, 0, 0))
            for x in pose:
                GL.glPushMatrix()
                GL.glTranslatef(x[0, 0], x[1, 0], -x[2, 0])
                GLUT.glutSolidSphere(2.e-2, 32, 32)
                GL.glPopMatrix()
    ## mouse press event handler
    # @param self The object pointer
    # @param event mouse event
    def mousePressEvent(self, event):
        self._last = event.pos()
    ## mouse move event handler
    # @param self The object pointer
    # @param event mouse event
    def mouseMoveEvent(self, event):
        if self._last is not None:
            dx = event.x() - self._last.x()
            dy = event.y() - self._last.y()
            if event.buttons() & QtCore.Qt.LeftButton:
                self._rot[0] += dy
                self._rot[1] += dx
        self._last = event.pos()
        self.updateGL()
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
