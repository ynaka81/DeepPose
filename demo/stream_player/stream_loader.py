## Base class of loading streams
#
# The base class of loading stream file
class StreamLoader(object):
    ## get iterator
    # @param self The object pointer
    # @return iterator
    def __iter__(self):
        return self
    ## get length
    # @param self The object pointer
    # @return lengh of the stream
    def __len__(self):
        raise NotImplementedError("StreamLoader.__len__(self) should be overrided.")
    ## get next item
    # @param self The object pointer
    # @return the next item
    def next(self):
        raise NotImplementedError("StreamLoader.next(self) should be overrided.")
