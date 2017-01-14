from PIL import Image
import numpy as np

from stream_player.stream_loader import StreamLoader

## Image sequence loader
#
# The stream loader of image sequence
class ImageSequenceLoader(StreamLoader):
    ## constructor
    # @param filenames The name of files
    def __init__(self, filenames):
        ## filenames
        self._filenames = filenames
        ## sequence index
        self._index = 0
    ## get length
    # @param self The object pointer
    # @return lengh of the stream
    def __len__(self):
        return len(self._filenames)
    ## get next item
    # @param self The object pointer
    # @return the next item
    def next(self):
        if self._index >= len(self._filenames):
            raise StopIteration()
        else:
            # read image as numpy array
            f = Image.open(self._filenames[self._index])
            try:
                image = np.asarray(f, dtype=np.uint8)
                if image.shape != (227, 227, 3):
                    raise ValueError("The image size {0} should be (227, 227, 3).".format(image.shape))
            finally:
                # cope with pillow < 3.0
                if hasattr(f, "close"):
                    f.close()
            self._index += 1
            return image
