from Video_Processing import *
from Colorization import *
from Color_Propagation.Contours_Propagation import *

VideoFileName = "Test Video/V1.avi"
FrameList = getVideoFrames(VideoFileName)
ColorizedFrameList = []  # Contain video frames after colorization
Model = LoadColorizationModel()  # Loading the colorization Model
isKeyFrame = True  # first frame in the scene only colorized
I0 = np.zeros((FrameList[0].shape[0], FrameList[0].shape[1]))  # First colorized Frame
FrameList = FrameList[70:]
for frame in FrameList:
    if isKeyFrame:
        I0 = colorization(frame, Model)
        # show_images([I0, frame], ["Colorized", "Original"])
        isKeyFrame = False
        ColorizedFrameList.append(I0)
    else:
        Gk = frame
        Ik_1 = ColorizedFrameList[-1]
        Gk_1 = rgb2gray(Ik_1)

        ColorizedFrameList.append(ContourPropagation(Gk, Gk_1, Ik_1))  # no propagation
        show_images([ColorizedFrameList[-1], frame], ["Colorized", "Original"])

        print("Generated Frame#")

# TODO: Collect the frames and output the video
# TODO: Link the voice of the input video with the colorized frames
