from commonfunctions import *
import cv2 as cv


def getVideoFrames(FileName):
    """This function return all the video or clip frames """
    print("[INFO] starting video stream...")
    # vs = VideoStream(src=0).start() if camera mode
    cap = cv.VideoCapture(FileName)
    hasFrame, frame = cap.read()
    frame = cv2.resize(frame, (720, 720))
    VideoFrames = []
    while hasFrame:
        # show_images([frame])
        VideoFrames.append(np.copy(frame))
        hasFrame, frame = cap.read()
        #if hasFrame:
           # frame = cv2.resize(frame, (100, 100))
    print("[INFO] video stream...done")
    return VideoFrames


def WriteFrames(FileName, image):
    """This function return all the video or clip frames """
    filename = 'Output/Frame#' + str(FileName) + '.png'
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
   # image = cv2.resize(image, (720, 1024))
    cv2.imwrite(filename, image)
    print("[INFO] Frame" + str(FileName) + " has been written...done")


def WriteVideo(ColorizedFrames):
    """"Suppose to Write a Video After processing the frames"""
    outputFile = args.input[:-4] + '_colorized.avi'
    vid_writer = cv.VideoWriter(outputFile, cv.VideoWriter_fourcc('M', 'J', 'P', 'G'), 60,
                                (frame.shape[1], frame.shape[0]))

# getVideoFrames("greyscaleVideo.mp4")
def sorted_nicely( l ):
    """ Sort the given iterable in the way that humans expect."""
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ]
    return sorted(l, key = alphanum_key)

def FrameCapture(path):
    # Path to video file
    vidObj = cv2.VideoCapture(path)

    # Used as counter variable
    count = 0

    # checks whether frames were extracted
    success = 1

    while success:
        # vidObj object calls read
        # function extract frames
        success, image = vidObj.read()

        #comment this line for colorized images
        #image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

        # Saves the frames with frame-count
        cv2.imwrite(r"Test Video\\ReSegemented\\frame%d.jpg" % count, image)
    #### WILL GIVE UNKNOWN ERROR BUT OUTPUTS CORRECTLY DONT WASTE UR TIME
        count += 1
