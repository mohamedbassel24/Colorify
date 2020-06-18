from Colorization.GAN import Load_GAN, colorization
from Video_Processing import *
from Color_Propagation.Contours_Propagation import ColorPropagation_ShootFrames

videoPath = "test_data/Videos/Little Lion is walking on the grass without shoes....mp4    "  # The Path of Image to be colorized
imgPath = "test_data/Images/park.jpg"  # The path of video to be colorized

VideoMode = 1  # 0: Image Colorization , 1:Video Colorization
ColorizationByFrame = 1
# Load Generator Model
gen_model = Load_GAN()
if VideoMode:
    # load movie Frames
    FrameList = getVideoFrames(videoPath)
    # List Contain movie frames after colorization
    ColorizedFrameList = []
    if not ColorizationByFrame:
        # cut the movie into shoots
        shootList = getFrameShoots(FrameList, Threshold=4500, showSteps=False)  # tune this Threshold for each video
        for i in range(len(shootList)):
            print("INFO : Shoot #", i + 1, "/", len(shootList), " is Processing .. ")
            # get the keyFrame: Frame that contains most of objects and return its index in the shootList
            keyFrame, indexKeyFrame = getKeyFrame(shootList[i])
            # Colorize the KeyFrame
            colorized_keyFrame = colorization(keyFrame, gen_model)
            # Propagate the color to the rest of shootFrames
            ColorizedFrameList += ColorPropagation_ShootFrames(shootList[i], colorized_keyFrame, indexKeyFrame, i)
    else:
        print("INFO: Colorizing Frame by Frame .. ")
        count_frame = 0  # counter for printing purpose
        for frame in FrameList:
            # displaying #Frame processing
            print("Frame #", count_frame, "/", len(FrameList), " is Processing .. ")
            # Colorize the frame and append it in colorization list
            ColorizedFrameList.append(colorization(frame, gen_model))
            # increment frame counter
            count_frame += 1
    # Integrate frames to make a complete movie
    WriteMovieFrames(ColorizedFrameList, "Output/OutputVideo")
    # Link Movie Audio with the Frames
    IntegrateAudio(videoPath, "Output/OutputVideo")
else:

    img = io.imread(imgPath)  # Read image
    img = colorization(img, gen_model)  # Colorize the image
    WriteImage(img)  # Write the image
