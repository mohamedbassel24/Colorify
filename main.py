from Colorization.GAN import Load_GAN, colorization
from Video_Processing import *
from Color_Propagation.Contours_Propagation import ColorPropagation_ShootFrames

videoPath = "test_data/Videos/sample.mp4"  # The Path of Image to be colorized
imgPath = "test_data/Images/frances-ha-2012-003-greta-gerwig.jpg"  # The path of video to be colorized

VideoMode = 0  # 0: Image Colorization , 1:Video Colorization
# Load Generator Model
gen_model = Load_GAN()
if VideoMode:
    # load movie Frames
    FrameList = getVideoFrames(videoPath)
    # cut the movie into shoots
    shootList = getFrameShoots(FrameList, Threshold=3000)  # tune this Threshold for each video
    # List Contain movie frames after colorization
    ColorizedFrameList = []
    for shootFrames in shootList:
        # get the keyFrame: Frame that contains most of objects and return its index in the shootList
        keyFrame, indexKeyFrame = getKeyFrame(shootFrames)
        # Colorize the KeyFrame
        colorized_keyFrame = colorization(keyFrame, gen_model)
        # Propagate the color to the rest of shootFrames
        ColorizedFrameList += ColorPropagation_ShootFrames(shootFrames, colorized_keyFrame, indexKeyFrame)
    # Integrate frames to make a complete movie
    WriteMovieFrames(ColorizedFrameList, "Output/OutputVideo")
    # Link Movie Audio with the Frames
    IntegrateAudio(videoPath, "Output/OutputVideo")
else:

    img = io.imread(imgPath)  # Read image
    img = colorization(img, gen_model)  # Colorize the image
    WriteImage(img)  # Write the image
