from Video_Processing import *
from Colorization import *
from Color_Propagation.Contours_Propagation import *

VideoFileName = "Test Video/greyscaleVideo.mp4"
FrameList = getVideoFrames(VideoFileName)
ColorizedFrameList = []  # Contain video frames after colorization
Model = LoadColorizationModel()  # Loading the colorization Model
isKeyFrame = True  # first frame in the scene only colorized
I0 = np.zeros((FrameList[0].shape[0], FrameList[0].shape[1]))  # First colorized Frame
FrameList = FrameList[100:]
preFrame = 0
FrameNum = 0
#TODO:
for frame in FrameList:
    if isKeyFrame:
        I0 = colorization(frame, Model)
        # show_images([I0, frame], ["Colorized", "Original"])
        isKeyFrame = False
        ColorizedFrameList.append(I0)
        preFrame = frame
        FrameNum += 1
    else:
        Gk = frame
        Ik_1 = ColorizedFrameList[-1]
        Gk_1 = preFrame
        # show_images([Ik_1, Gk_1, Gk], ["Colorized Pre", "pre frame", "curr frame"])

        ColorizedFrameList.append(ContourPropagation(Gk, Gk_1, Ik_1))  # no propagation
        # show_images([ColorizedFrameList[-1], frame], ["Colorized", "Original"])
        WriteFrames(FrameNum, ColorizedFrameList[-1])
        preFrame = frame
        FrameNum += 1
# TODO: Collect the frames and output the video
# TODO: Link the voice of the input video with the colorized frames
