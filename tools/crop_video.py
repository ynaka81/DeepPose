import argparse
import os
import sys
import cv2

## Crop a video
#
# The video cropping tool for Deep3dPose demo
class CropVideo(object):
    ## constructor
    # @param args The command line arguments
    def __init__(self, args):
        self.args = args
    ## main method of cropping the video
    # @param self The object pointer
    def main(self):
        args = self.args
        # make output directory
        path = os.path.join(args.out, os.path.splitext(os.path.basename(args.file))[0])
        try:
            os.makedirs(path)
        except OSError:
            pass
        # parse arguments
        u0, v0 = [int(v) for v in args.start.split(",")]
        cx, cy = [int(v) for v in args.crop.split(",")]
        # load video
        frame = 0
        video = cv2.VideoCapture(args.file)
        max_frame = int(video.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))
        while(video.isOpened()):
            # get current image frame
            ret, image = video.read()
            if not ret:
                break
            # resize and crop image
            h, w, _ = image.shape
            image = cv2.resize(image, None, fx=args.resize, fy=args.resize)
            image = image[v0:v0 + cy, u0:u0 + cx, :]
            # preview or save image
            if args.preview:
                # preview image
                cv2.imshow("preview", image)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            else:
                # save image
                filename = os.path.join(path, "{0}.png".format(frame))
                cv2.imwrite(filename, image)
                sys.stderr.write("generating... frames({0}/{1})\r".format(frame, max_frame))
                sys.stderr.flush()
                # increment image frame
                frame += 1
        # post processing
        video.release()
        if args.preview:
            cv2.destroyAllWindows()
        else:
            sys.stderr.write("\n")

if __name__ == "__main__":
    # arg definition
    parser = argparse.ArgumentParser(description="Crop a video for Deep3dPose demo.")
    parser.add_argument("file", type=str, help="Video name to crop")
    parser.add_argument("--start", "-s", type=str, default="0,0", help="Start point to crop")
    parser.add_argument("--resize", "-r", type=float, default=1.0, help="Resize scale")
    parser.add_argument("--crop", "-c", type=str, default="227,227", help="Crop size")
    parser.add_argument("--out", "-o", type=str, default="data/demo_images", help="Output directory")
    parser.add_argument("--preview", action="store_true", default=False, help="True when you would preview")
    CropVideo(parser.parse_args()).main()
