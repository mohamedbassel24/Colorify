from Video_Processing import *
from Colorization import *
from Color_Propagation import *

VideoFileName = "greyscaleVideo.mp4"
FrameList = getVideoFrames(VideoFileName)  # TODO:NOT real time as i get all the frames ?
ColorizedFrameList = []  # Contain video frames after colorization
Model = LoadColorizationModel()  # Loading the colorization Model
isKeyFrame = True  # first frame in the scene only colorized
I0 = np.zeros((FrameList[0].shape[0], FrameList[0].shape[1]))  # First colorized Frame
for frame in FrameList:
    if isKeyFrame:
        I0 = colorization(frame, Model)
        # show_images([I0, frame], ["Colorized", "Original"])
        isKeyFrame = False
        ColorizedFrameList.append(I0)
    else:
        # 4 Parameters => ColorizedKeyFrame,CurrentGrey,PreviousGrey,PreviousColorized

        Gk = frame
        Ik_1 = ColorizedFrameList[-1]
        Gk_1 = rgb2gray(Ik_1)

        ColorizedFrameList.append(colorization(frame, Model)) # no propagation
        show_images([ColorizedFrameList[-1], frame], ["Colorized", "Original"])
        # Color propagation
        # Method-1: Segmentation
        # Method-2: Style Transfer
        # Method-3: local and global propagation optical flow

        print("empty")

# TODO: Collect the frames and output the video
# TODO: Link the voice of the input video with the colorized frames
