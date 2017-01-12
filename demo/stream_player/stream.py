## Stream structure
#
# The structure class of stream
class Stream(object):
    ## constructor
    # @param frame The current frame
    # @param image The image of the current frame
    # @param pose The 3D pose of the current frame
    def __init__(self, frame, image, pose):
        for k, v in locals().items():
            setattr(self, k, v)
