import cv2
import glob
import scenedetect
from scenedetect.video_manager import VideoManager
from scenedetect.scene_manager import SceneManager
from scenedetect.frame_timecode import FrameTimecode
from scenedetect.stats_manager import StatsManager
from scenedetect.detectors import ContentDetector


def extractScenes(pathIn, videoStart=20, videoEnd=120.0, scenesThreshold=15.0):
    """ This Function Extract scenes from a grayscale video given its path and video begin and end (optional)"""
    video_manager = VideoManager([pathIn])
    stats_manager = StatsManager()
    scene_manager = SceneManager(stats_manager)
    # Add ContentDetector algorithm (constructor takes detector options like threshold).
    scene_manager.add_detector(ContentDetector(threshold=scenesThreshold))
    base_timecode = video_manager.get_base_timecode()

    try:

        start_time = base_timecode + videoStart  # 00:00:00.667
        end_time = base_timecode + videoEnd  # 00:00:60.000
        # Set video_manager duration to read frames from 00:00:00 to 00:00:20.
        video_manager.set_duration(start_time=start_time, end_time=end_time)

        # Set downscale factor to improve processing speed (no args means default).
        video_manager.set_downscale_factor()

        # Start video_manager.
        video_manager.start()

        # Perform scene detection on video_manager.
        scene_manager.detect_scenes(frame_source=video_manager)

        # Obtain list of detected scenes.
        scene_list = scene_manager.get_scene_list(base_timecode)
        # Like FrameTimecodes, each scene in the scene_list can be sorted if the
        # list of scenes becomes unsorted.

        print('List of scenes obtained:')
        for i, scene in enumerate(scene_list):
            print("File Name " + pathIn + '    Scene %2d: Start %s / Frame %d, End %s / Frame %d' % (
                i + 1, scene[0].get_timecode(), scene[0].get_frames(), scene[1].get_timecode(), scene[1].get_frames(),))
    finally:
        video_manager.release()
    return scene_list


def extractImages(pathIn, IsRotated):
    """ Reading the frames for each second """
    count = 0
    vidcap = cv2.VideoCapture(pathIn)
    success, image = vidcap.read()
    success = True
    ListFrames = []
    while success:
        # save a frame for every second
        vidcap.set(cv2.CAP_PROP_POS_MSEC, (count * 1000))
        success, image = vidcap.read()
        #   print('Read a new frame: ', success)
        if success == 0:
            break
        if IsRotated:
            image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        ListFrames.append(image)
        count = count + 1
    return ListFrames


for filename in sorted(glob.glob('../02-Dataset/Training_Data/*.mp4')):
    # filename = "video1.mp4"  # Test specific video
    VideoFrames = []  # Representing the whole video frames
    ListOfScenes = []  # Representing the list that hold each separate scene
    VideoStart = 0  # Specify the start of the video for scene detection
    VideoEnd = 120.0  # Specify the end of the video for scene detection
    ScenesThreshold = 15  # Specify the minimum change between scenes pixels intensity for more or less scenes
    ListOfScenes = extractScenes(filename, VideoStart, VideoEnd,
                                 ScenesThreshold)  # get the start and end for each scene
    VideoFrames = extractImages(filename, False)  # Get All the video frames
    for Scene in ListOfScenes:
        SceneStart = Scene[0]  # Representing the end of the scene
        SceneEnd = Scene[1]  # Representing the end of the scene
        print(len(VideoFrames))
        print("Start: " + str(SceneStart) + "End: " + str(SceneEnd))
        ListOfFrames = []  # Representing the frames for each scene
        # TODO: Get the MI Frame from List of Frames
