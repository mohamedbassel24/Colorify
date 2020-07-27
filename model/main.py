from Colorization.GAN import Load_GAN_Human, Load_GAN_Nature, colorization
from Video_Processing import *
from Color_Propagation.Contours_Propagation import ColorPropagation_ShootFrames, Interactive

# videoPath = "test_data/Videos/picnic.avi"  # The Path of Image to be colorized
# imgPath = "test_data/Images/park.jpg"  # The path of video to be colorized

videoPath = "Input_and_Output/" + sys.argv[3]  # The Path of Image to be colorized
imgPath = "Input_and_Output/" + sys.argv[3]  # The path of video to be colorized
VideoMode = int(sys.argv[1])  # 0: Image Colorization , 1:Video Colorization
ColorizationByFrame = 0
# Load Generator Model
if int(sys.argv[2]) == 0:
    gen_model = Load_GAN_Human()  # load human model
else:
    gen_model = Load_GAN_Nature()  # load nature model

if VideoMode:
    # load movie Frames
    FrameList = getVideoFrames(videoPath)
    # List Contain movie frames after colorizationFailed to load resource: the server responded with a status of 403 ()

    ColorizedFrameList = []
    if not ColorizationByFrame:
        # cut the movie into shoots
        shootList = getFrameShoots(FrameList, Threshold=8000, showSteps=False)  # tune this Threshold for each video
        for i in range(len(shootList)):
            print("[INFO] : Shoot #", i + 1, "/", len(shootList), " is Processing .. ")
            # get the keyFrame: Frame that contains most of objects or Frame that contains most of colorized objects
            keyFrame, indexKeyFrame = getKeyFrame(shootList[i],
                                                  gen_model, Type=0)  # depends on the key type and return its index
            # Colorize the KeyFrame
            colorized_keyFrame = colorization(keyFrame, gen_model)
            # Interaction with the use to colorize the keyframe
            colorized_keyFrame = Interactive(colorized_keyFrame)
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
    WriteMovieFrames(ColorizedFrameList, "Input_and_Output/OutputVideo")
    # Link Movie Audio with the Frames
    IntegrateAudio(videoPath, "Input_and_Output/OutputVideo")

else:

    img = io.imread(imgPath)  # Read image
    img = colorization(img, gen_model)  # Colorize the image
    WriteImage(img)  # Write the image
