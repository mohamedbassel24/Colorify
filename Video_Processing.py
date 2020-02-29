from commonfunctions import *
import cv2 as cv


def getVideoFrames(FileName):
    """This function return all the video or clip frames """
    print("[INFO] starting video stream...")
    # vs = VideoStream(src=0).start() if camera mode
    cap = cv.VideoCapture(FileName)
    hasFrame, frame = cap.read()
    VideoFrames = []
    while hasFrame:
        # show_images([frame])
        VideoFrames.append(np.copy(frame))
        hasFrame, frame = cap.read()
    print("[INFO] video stream...done")
    return VideoFrames


def WriteVideo(ColorizedFrames):
    """"Suppose to Write a Video After processing the frames"""
    outputFile = args.input[:-4] + '_colorized.avi'
    vid_writer = cv.VideoWriter(outputFile, cv.VideoWriter_fourcc('M', 'J', 'P', 'G'), 60,
                                (frame.shape[1], frame.shape[0]))

#getVideoFrames("greyscaleVideo.mp4")
