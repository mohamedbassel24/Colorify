from commonfunctions import *
from skimage import io, color
from scipy import stats


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


def ContourPropagation(gk, gk_prev, ik_pre):
    """ Propagate color through img contours"""

    # Convert RGB to LAB color model
    ik_pre = color.rgb2lab(ik_pre)
    # initialize the output color image
    ik = np.zeros(ik_pre.shape, dtype="float64")  # lab is a float data type
    # get the GrayScale of current frame and reformative it in range of 0 99
    ik[:, :, 0] = (rgb2gray(gk_prev)) * 100
    # Convert to Binary Image

    gk = (rgb2gray(gk) * 255).astype("uint8")
    gk_prev = (rgb2gray(gk_prev) * 255).astype("uint8")
    _, gk = cv2.threshold(gk, 100, 255, cv2.THRESH_BINARY)
    _, gk_prev = cv2.threshold(gk_prev, 100, 255, cv2.THRESH_BINARY)

    # TODO:MOrplogical operation here

    # Get Image Contours
    contours_gk, _ = cv2.findContours(gk, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  # for current frame
    contours_gk_pre, _ = cv2.findContours(gk_prev, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  # for previous frame

    # Match Contours
    for cnt_Gk in contours_gk:
        MatchedGkPre = cnt_Gk
        center_cntGK = getContourAverage(gk, cnt_Gk)
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
            if matchingRatio < 0.01:
                MinDis = D2Center  # get the minimum distance
                IndexToRemove = cnt_Index
                MatchedGkPre = contours_gk_pre[cnt_Index]
                break
        # remove the last pick from list
        if len(contours_gk_pre) != 0:
            contours_gk_pre.pop(IndexToRemove)

        # GET  the points of the contour
        gk_pointsCnt = getContourPoints(gk, cnt_Gk)
        gk_pre_pointsCnt = getContourPoints(gk_prev, MatchedGkPre)

        [cX, cY] = getContourAverage(gk_prev, MatchedGkPre)
        # casting from nparray to list
        NewPoints = []
        # getting the pixels that are outer of the matched contour
        Mask_a = ik_pre[gk_pre_pointsCnt[0], gk_pre_pointsCnt[1], 1]  # pixels of the contour
        Mask_b = ik_pre[gk_pre_pointsCnt[0], gk_pre_pointsCnt[1], 2]
        mode_a = get_modeArr(Mask_a)  # mode of channel a
        mode_b = get_modeArr(Mask_b)  # mode of channel b

        # Resizing the 2 list points
        print(len(gk_pointsCnt[0]), len(gk_pre_pointsCnt[0]))
        if len(gk_pointsCnt[0]) < len(gk_pre_pointsCnt[0]):
            gk_pre_pointsCnt[0] = gk_pre_pointsCnt[0][:len(gk_pointsCnt[0])]
            gk_pre_pointsCnt[1] = gk_pre_pointsCnt[1][:len(gk_pointsCnt[1])]
        elif len(gk_pointsCnt[0]) > len(gk_pre_pointsCnt[0]):
            NewPoints = [gk_pointsCnt[0][len(gk_pre_pointsCnt[0]):], gk_pointsCnt[1][len(gk_pre_pointsCnt[0]):]]
            gk_pointsCnt[0] = gk_pointsCnt[0][:len(gk_pre_pointsCnt[0])]
            gk_pointsCnt[1] = gk_pointsCnt[1][:len(gk_pre_pointsCnt[1])]

        # Shifting the colors from previous to current
        if len(gk_pointsCnt[0]) > 1:
            ik[gk_pointsCnt[0], gk_pointsCnt[1], 1] = ik_pre[gk_pre_pointsCnt[0], gk_pre_pointsCnt[1], 1]  # a channel
            ik[gk_pointsCnt[0], gk_pointsCnt[1], 2] = ik_pre[gk_pre_pointsCnt[0], gk_pre_pointsCnt[1], 2]  # b channel

        if len(NewPoints) > 1:
            ik[NewPoints[:][0], NewPoints[:][1], 1] = mode_a  # a channel
            ik[NewPoints[:][0], NewPoints[:][1], 2] = mode_b  # b channel
    # Convert to RGB color model

    ik = color.lab2rgb(ik)
    ik = (ik * 255).astype("uint8")

    # Use a median filter to overcome some grayscale pixels
    # ik = cv2.bilateralFilter(ik, 300, 20, 100)
    return ik
