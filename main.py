from Video_Processing import *
from Colorization import *
from Color_Propagation.Contours_Propagation import *

VideoFileName = "Test Video/V2.avi"
FrameList = getVideoFrames(VideoFileName)
ColorizedFrameList = []  # Contain video frames after colorization
Model = LoadColorizationModel()  # Loading the colorization Model
isKeyFrame = True  # first frame in the scene only colorized
I0 = np.zeros((FrameList[0].shape[0], FrameList[0].shape[1]))  # First colorized Frame
FrameList = FrameList[1:]
preFrame = 0
FrameNum = 0
# print(np.sum(abs(FrameList[2] - FrameList[1])))
shootList = getFrameShoots(FrameList,Threshold=300)
for shootFrames in shootList:
    # get the keyFrame: Frame that contains all objects
    keyFrame, indexKeyFrame = getKeyFrame(shootFrames)
    I0 = colorization(keyFrame, Model)
    ColorizedFrameList.append(I0)
    # Forward Propagation
    for i in range(indexKeyFrame + 1, len(shootFrames),1):
        Gk = shootFrames[i]
        Ik_1 = ColorizedFrameList[-1]
        Gk_1 = shootFrames[i - 1]
    #    show_images([Ik_1, Gk_1, Gk], ["Colorized Pre", "pre frame", "curr frame"])
        ColorizedFrameList.append(ContourPropagation(Gk, Gk_1, Ik_1))
        # show_images([ColorizedFrameList[-1], frame], ["Colorized", "Original"])
        WriteFrames(FrameNum, ColorizedFrameList[-1])
        FrameNum += 1
    # Backward Propagation
    for i in range(indexKeyFrame - 1, 0, -1):

        Gk = shootFrames[i]
        Ik_1 = ColorizedFrameList[0]
        Gk_1 = shootFrames[i + 1]
        show_images([Ik_1, Gk_1, Gk], ["Colorized Pre", "post frame", "curr frame"])
        ColorizedFrameList.insert(0,ContourPropagation(Gk, Gk_1, Ik_1))  # insert at beginning
        WriteFrames(FrameNum, ColorizedFrameList[0])
        FrameNum += 1
# TODO: Link the voice of the input video with the colorized frames and write it 
