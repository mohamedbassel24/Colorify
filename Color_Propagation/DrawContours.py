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


def ContourPropagation2(gk, gk_prev, ik_pre):
    """ Propagate color through img contours"""

    # initialize the output color image
    ik = np.zeros(ik_pre.shape, dtype="uint8")  # lab is a float data type
    # get the GrayScale of current frame and reformative it in range of 0 99
    # Convert to Binary Image

    gk = (rgb2gray(gk) * 255).astype("uint8")
    gk_prev = (rgb2gray(gk_prev) * 255).astype("uint8")
    GlobalThresh = threshold_otsu(gk)

    #  GlobalThresh = 111
    GlobalThresh = 127
    _, gk = cv2.threshold(gk, GlobalThresh, 255, cv2.THRESH_BINARY)
    _, gk_prev = cv2.threshold(gk_prev, GlobalThresh, 255, cv2.THRESH_BINARY)

    # Get Image Contours
    contours_gk_pre, _ = cv2.findContours(gk, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  # for current frame
    contours_gk, _ = cv2.findContours(gk_prev, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  # for previous frame

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
            #   print(D2Center)
            if matchingRatio < 0.01 and D2Center < 10:
                MinDis = D2Center  # get the minimum distance
                IndexToRemove = cnt_Index
                MatchedGkPre = contours_gk_pre[cnt_Index]
                break
        # remove the last pick from list
        if len(contours_gk_pre) != 0:
            contours_gk_pre.pop(IndexToRemove)

        # GET  the points of the contour
        mask = np.zeros(gk.shape, np.uint8)
        center_match = getContourAverage(gk_prev, MatchedGkPre)  # g
        mask[...] = 0
        cv2.drawContours(mask, [MatchedGkPre], 0, 255, -1)
        mColor = cv2.mean(ik_pre, mask)
        mColor = list(mColor)
        mColor[:3] = ik_pre[center_match[0], center_match[1]]
        mColor = tuple(mColor)

        moment = cv2.moments(MatchedGkPre)
        c_y = moment['m10'] / (moment['m00'] + 0.01)
        c_x = moment['m01'] / (moment['m00'] + 0.01)
        c_y = int(c_y)
        c_x = int(c_x)
        centroid_color = ik_pre[c_x, c_y]
        centroid_color = tuple ([int(x) for x in centroid_color])
        cv2.drawContours(ik, [cnt_Gk], 0, centroid_color, -1)
        # show_images([ik, mask])

   # show_images([ik])

    return ik





