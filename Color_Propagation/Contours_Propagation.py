from commonfunctions import *
from skimage import io, color
from scipy import stats
from Video_Processing import WriteFrames


def getContourCenter(cnt):
    """get Center of specific contour"""
    c = max(cnt, key=cv2.contourArea)
    M = cv2.moments(cnt)
    if M["m00"] != 0:
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])

    else:
        cX, cY = 0, 0
    return [cX, cY]


def getContourAverage(img, cnt):
    pnts = getContourPoints(img, cnt)
    """get Center of specific contour"""
    x = 0
    y = 0

    for kp in range(len(pnts[0])):  # outer points
        x = x + pnts[0][kp]
        y = y + pnts[1][kp]
    if len(pnts[0]) == 0:
        return 0, 0
    return [np.uint8(np.ceil(x / len(pnts[0]))), np.uint8(np.ceil(y / len(pnts[0])))]


def getContourPoints(img, cnt):
    """Get all the Pixels that belong to specific contour """
    mask = np.zeros(img.shape, dtype="uint8")
    mask = cv2.drawContours(mask, [cnt], 0, color=255, thickness=-1)  # create a mask for a points belong to contour
    pnts = np.where(mask == 255)
    pnts = list(pnts)
    pnts[0] = list(pnts[0])
    pnts[1] = list(pnts[1])
    return pnts


def get_mode(img):
    unq, count = np.unique(img.reshape(-1, img.shape[-1]), axis=0, return_counts=True)
    return unq[count.argmax()]


def get_modeArr(Arr):
    mode_info = stats.mode(Arr)
    return mode_info[0]


def ContourPropagation(gk, gk_prev, ik_pre, ShowSteps=False, BinaryThreshold=103, UseMode=False):
    """ Propagate color through img contours """
    # ==> resizing
    # Get the original shape of the colorized pre frame image
    originalShape = ik_pre.shape
    # Copy the current grayscale image
    original_gk = np.copy(gk)
    # Resize the pre colorized frame
    ik_pre = cv2.resize(ik_pre, (256, 256), interpolation=cv2.INTER_AREA)
    # Resize pre grayscale frame
    gk_prev = cv2.resize(gk_prev, (256, 256), interpolation=cv2.INTER_AREA)
    # Resize current grayscale frame
    gk = cv2.resize(gk, (256, 256), interpolation=cv2.INTER_AREA)
    # -----------------------------------------------------------------------------------
    # Convert RGB to LAB color model
    ik_pre = color.rgb2lab(ik_pre)
    # initialize the output color image
    ik = np.zeros(ik_pre.shape, dtype="float64")  # lab is a float data type
    # get the GrayScale of current frame and reformative it in range of 0 99
    ik[:, :, 0] = (rgb2gray(gk)) * 100
    # Create a map to track each pixel
    PixelMap = np.zeros((ik_pre.shape[0], ik_pre.shape[1]))
    # Convert to Binary Image

    gk = (rgb2gray(gk) * 255).astype("uint8")
    gk_prev = (rgb2gray(gk_prev) * 255).astype("uint8")
    GlobalThresh = 200
    # GlobalThresh = threshold_otsu(gk)
    # GlobalThresh = 160
    # GlobalThresh = BinaryThreshold
    #  GlobalThresh = 111
    # GlobalThresh = 103

    _, gk = cv2.threshold(gk, GlobalThresh, 255, cv2.THRESH_BINARY)
    _, gk_prev = cv2.threshold(gk_prev, GlobalThresh, 255, cv2.THRESH_BINARY)
    if ShowSteps:
        show_images([gk, gk_prev])
    # Get Image Contours
    contours_gk, _ = cv2.findContours(gk, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  # for current frame
    contours_gk_pre, _ = cv2.findContours(gk_prev, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  # for previous frame

    # Match Contours

    for cnt_Gk in contours_gk:  # for each contour in the current frame
        MatchedGkPre = cnt_Gk
        center_cntGK = getContourAverage(gk, cnt_Gk)  # get the Center of mass for each contour
        # finding the corresponding contour in previous frame
        MinDis = 100
        IndexToRemove = 0
        for cnt_Index in range(len(contours_gk_pre)):
            matchingRatio = cv2.matchShapes(cnt_Gk, contours_gk_pre[cnt_Index], 1,
                                            0.0)  # get the probability of matching
            center_match = getContourAverage(gk_prev, contours_gk_pre[cnt_Index])  # get center of cnt
            # get distance between 2 contour centers
            D2Center = np.sqrt(
                (int(center_cntGK[0]) - int(center_match[0])) ** 2 + (int(center_cntGK[1]) - int(center_match[1])) ** 2)
            #   print(D2Center)
            if matchingRatio < 0.01 and D2Center < 10:  # it was found that 0.01 and 10 are suitable for matching
                MinDis = D2Center  # get the minimum distance
                IndexToRemove = cnt_Index
                MatchedGkPre = contours_gk_pre[cnt_Index]
                if ShowSteps:
                    print("Match Counters with Distance  ", D2Center, "Matching Ratio :", matchingRatio)
                break

        # remove the picked counter from list to avoid matching it again
        if len(contours_gk_pre) != 0:
            contours_gk_pre.pop(IndexToRemove)

        # GET  the points of the contour
        gk_pointsCnt = getContourPoints(gk, cnt_Gk)  # points from current frame
        gk_pre_pointsCnt = getContourPoints(gk_prev, MatchedGkPre)  # points from pre frame

        [cX, cY] = getContourAverage(gk_prev, MatchedGkPre)

        # getting the pixels that are outer of the matched contour
        Mask_a = ik_pre[gk_pointsCnt[0], gk_pointsCnt[1], 1]  # pixels of the contour
        Mask_b = ik_pre[gk_pointsCnt[0], gk_pointsCnt[1], 2]
        mode_a = get_modeArr(Mask_a)  # mode of channel a
        mode_b = get_modeArr(Mask_b)  # mode of channel b

        # Resizing the 2 list points
        if len(gk_pointsCnt[0]) < len(gk_pre_pointsCnt[0]):
            gk_pre_pointsCnt[0] = gk_pre_pointsCnt[0][:len(gk_pointsCnt[0])]
            gk_pre_pointsCnt[1] = gk_pre_pointsCnt[1][:len(gk_pointsCnt[1])]
            if UseMode:
                ik[gk_pointsCnt[0], gk_pointsCnt[1], 1] = mode_a  # a channel
                ik[gk_pointsCnt[0], gk_pointsCnt[1], 2] = mode_b  # b channel
            else:
                ik[gk_pointsCnt[0], gk_pointsCnt[1], 1] = ik_pre[
                    gk_pre_pointsCnt[0], gk_pre_pointsCnt[1], 1]  # a channel
                ik[gk_pointsCnt[0], gk_pointsCnt[1], 2] = ik_pre[
                    gk_pre_pointsCnt[0], gk_pre_pointsCnt[1], 2]  # b channel
        elif len(gk_pointsCnt[0]) > len(gk_pre_pointsCnt[0]):
            ik[gk_pointsCnt[0][len(gk_pre_pointsCnt[0]):], gk_pointsCnt[1][
                                                           len(gk_pre_pointsCnt[0]):], 1] = mode_a  # a channel
            ik[gk_pointsCnt[0][len(gk_pre_pointsCnt[0]):], gk_pointsCnt[1][
                                                           len(gk_pre_pointsCnt[0]):], 2] = mode_b  # b channel
            gk_pointsCnt[0] = gk_pointsCnt[0][:len(gk_pre_pointsCnt[0])]
            gk_pointsCnt[1] = gk_pointsCnt[1][:len(gk_pre_pointsCnt[1])]
            if UseMode:
                ik[gk_pointsCnt[0], gk_pointsCnt[1], 1] = mode_a  # a channel
                ik[gk_pointsCnt[0], gk_pointsCnt[1], 2] = mode_b  # b channel
            else:
                ik[gk_pointsCnt[0], gk_pointsCnt[1], 1] = ik_pre[
                    gk_pre_pointsCnt[0], gk_pre_pointsCnt[1], 1]  # a channel
                ik[gk_pointsCnt[0], gk_pointsCnt[1], 2] = ik_pre[
                    gk_pre_pointsCnt[0], gk_pre_pointsCnt[1], 2]  # b channel
        else:
            if UseMode:
                ik[gk_pointsCnt[0], gk_pointsCnt[1], 1] = mode_a  # a channel
                ik[gk_pointsCnt[0], gk_pointsCnt[1], 2] = mode_b  # b channel
            else:
                ik[gk_pointsCnt[0], gk_pointsCnt[1], 1] = ik_pre[
                    gk_pre_pointsCnt[0], gk_pre_pointsCnt[1], 1]  # a channel
                ik[gk_pointsCnt[0], gk_pointsCnt[1], 2] = ik_pre[
                    gk_pre_pointsCnt[0], gk_pre_pointsCnt[1], 2]  # b channel
                # assign the points for each contour in the global map
        PixelMap[gk_pointsCnt[0], gk_pointsCnt[1]] = 1
    # propagate the pixels with no contours
    ik[PixelMap == 0, 1] = ik_pre[PixelMap == 0, 1]
    ik[PixelMap == 0, 2] = ik_pre[PixelMap == 0, 2]

    # Convert to RGB color model

    ik = color.lab2rgb(ik)
    ik = (ik * 255).astype("uint8")

    ik = cv2.resize(ik, (originalShape[1], originalShape[0]), interpolation=cv2.INTER_AREA)
    #   ik = color.rgb2lab(ik)
    #  ik = ik.astype("float64")
    # ik[:, :, 0] = (rgb2gray(original_gk)) * 100
    # ik = color.lab2rgb(ik)
    # ik = (ik * 255).astype("uint8")

    return ik


def ColorPropagation_ShootFrames(shootFrames, keyFrame, indexKeyFrame, ShootNum):
    # Creating a directory for a processing shoot
    try:
        os.mkdir("Input_and_Output/Shoot#" + str(ShootNum + 1))
    except OSError as error:
        # Avoid error if the directory already created
        pass
        # List of All colorized Frames
    ColorizedFrameList = [keyFrame]
    # Forward Propagation from index frame to the end of the frame list
    for i in range(indexKeyFrame + 1, len(shootFrames), 1):
        # Current GrayScale Image
        Gk = shootFrames[i]
        # Previous Colorized Frame
        Ik_1 = ColorizedFrameList[-1]
        # Previous GrayScale Image
        Gk_1 = shootFrames[i - 1]
        # Check if the frame already a black image
        IsBlackImage = np.sum(Gk)  # get the sum in current frame
        # To avoid find contour null hierarchy error
        if IsBlackImage == 0:
            # avoid colorizing this frame
            ColorizedFrameList.append(Gk)
            continue
        ColorizedFrameList.append(ContourPropagation(Gk, Gk_1, Ik_1))
        # show_images([ColorizedFrameList[-1], frame], ["Colorized", "Original"])
        WriteFrames(i, ColorizedFrameList[-1], ShootNum + 1)
    # Backward Propagation from index frame to the start of the frame list
    for i in range(indexKeyFrame - 1, 0, -1):
        # Current GrayScale Image
        Gk = shootFrames[i]
        # Previous Colorized Frame
        Ik_1 = ColorizedFrameList[0]
        # Previous GrayScale Image
        Gk_1 = shootFrames[i + 1]
        # Check if the frame already a black image
        IsBlackImage = np.sum(Gk)  # get the sum in current frame
        # To avoid find contour null hierarchy error
        if IsBlackImage == 0:
            # avoid colorizing this frame
            ColorizedFrameList.append(Gk)
            continue
        ColorizedFrameList.insert(0, ContourPropagation(Gk, Gk_1, Ik_1))  # insert at beginning
        WriteFrames(i, ColorizedFrameList[0], ShootNum + 1)
    print("[INFO] Color Propagation in shoot# ", ShootNum + 1, "is done...")
    return ColorizedFrameList
