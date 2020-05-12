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
        VideoFrames.append(np.copy((frame)))
        hasFrame, frame = cap.read()
        # if hasFrame:
        # frame = cv2.resize(frame, (100, 100))
    print("[INFO] video stream...done")
    return VideoFrames


def WriteFrames(FileName, image):
    """This function return all the video or clip frames """
    filename = 'Output/' + str(FileName) + '.Frame.png'
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
def sorted_nicely(l):
    """ Sort the given iterable in the way that humans expect."""
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum_key)


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

        # comment this line for colorized images
        # image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

        # Saves the frames with frame-count
        cv2.imwrite(r"Test Video\\ReSegemented\\frame%d.jpg" % count, image)
        #### WILL GIVE UNKNOWN ERROR BUT OUTPUTS CORRECTLY DONT WASTE UR TIME
        count += 1


def getFrameShoots(FrameList, Threshold=500):
    """takes a clip and cut it into shoots based on the grayscale and average difference """
    preDifference = 0
    shootList = []
    shootFrames = []
    for i in range(len(FrameList) - 1):
        shootFrames.append(FrameList[i])
        currDiff = np.sum(abs(rgb2gray(FrameList[i + 1]) - rgb2gray(FrameList[i])))
        #  print(abs(currDiff - preDiffrence))
        shootDet = abs(currDiff - preDifference)
        if shootDet > Threshold:
            # for i in range(len(shootFrames)):
            # show_images([shootFrames[0], shootFrames[-1]], ["Start", "End"])
            shootList.append(shootFrames)
            shootFrames = []
            # print(shootDet)
        preDifference = currDiff
    return shootList


def getKeyFrame(rShoot):
    MaxContourNumber = 0
    KeyFrame = rShoot[0]
    indexKeyFrame = 0
    for i in range(len(rShoot)):
        gk = (rgb2gray(rShoot[i]) * 255).astype("uint8")
        _, gk = cv2.threshold(gk, 230, 255, cv2.THRESH_BINARY)
        # Get Image Contours
        contours_gk, _ = cv2.findContours(gk, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  # for current frame
        if len(contours_gk) > MaxContourNumber:
            KeyFrame = rShoot[i]
            indexKeyFrame = i
    return KeyFrame, indexKeyFrame


def FramesToVideo(dirr):
    img_array = []
    size = 0
    for filename in sorted(glob.glob(dirr)):
        print(filename)
        img = cv2.imread(filename)
        height, width, layers = img.shape
        size = (width, height)
        img_array.append(img)

    out = cv2.VideoWriter('project.avi', cv2.VideoWriter_fourcc(*'DIVX'), 15, size)

    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()
