from commonfunctions import *
from skimage import io, color


# Color propagation
# Method-1: Segmentation
# Method-2: Style Transfer
# Method-3: local and global propagation optical flow
def getContourCenter(cnt):
    """get Center of specific contour"""
    c = max(cnt, key=cv2.contourArea)
    M = cv2.moments(c)
    if M["m00"] != 0:
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        print("wee")
    else:
        cX, cY = 0, 0
    return [cX, cY]


def getContourAverage(img, cnt):
    _, pnts = getContourPoints(img, cnt)
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
    ListPoint = []
    for i in range(len(pnts[0])):
        ListPoint.append([pnts[0][i], pnts[1][i]])

    return ListPoint, pnts


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
            ret = cv2.matchShapes(cnt_Gk, contours_gk_pre[cnt_Index], 1, 0.0)  # get the probability of matching
            center_match = getContourAverage(gk_prev, contours_gk_pre[cnt_Index])  # get center of cnt
            # get distance between 2 contour centers
            D2Center = np.sqrt(
                (int(center_cntGK[0]) - int(center_match[0])) ** 2 + (int(center_cntGK[1]) - int(center_match[1])) ** 2)
            if ret < 0.1 and D2Center < MinDis:
                MinDis = D2Center  # get the minimum distance
                IndexToRemove = cnt_Index
                MatchedGkPre = contours_gk_pre[cnt_Index]
        # remove the last pick from list
        if len(contours_gk_pre) != 0:
            contours_gk_pre.pop(IndexToRemove)

        # if probability is close to 1 after looping => this is a new object => ignore it

        # GET  the points of the contour
        gk_pnts_list, gk_pt = getContourPoints(gk, cnt_Gk)
        gk_pre_pnts_list, gk_pre_pts = getContourPoints(gk_prev, MatchedGkPre)

        [cX, cY] = getContourAverage(gk_prev, MatchedGkPre)
        # casting from nparray to list
        gk_pre_pts = list(gk_pre_pts)
        gk_pre_pts[0] = list(gk_pre_pts[0])
        gk_pre_pts[1] = list(gk_pre_pts[1])

        # Resizing the 2 list points
        if len(gk_pnts_list) < len(gk_pre_pnts_list):
            gk_pre_pts = list(gk_pre_pts)
            gk_pre_pts[0] = gk_pre_pts[0][:len(gk_pt[0])]
            gk_pre_pts[1] = gk_pre_pts[1][:len(gk_pt[1])]
            gk_pre_pnts_list = gk_pre_pnts_list[:len(gk_pnts_list)]
        elif len(gk_pnts_list) > len(gk_pre_pnts_list):
            for i in range(len(gk_pnts_list) - len(gk_pre_pnts_list)):
                gk_pre_pnts_list.append([cX, cY])
                gk_pre_pts[0].append(cX)
                gk_pre_pts[1].append(cY)

        if len(gk_pt[0]) > 1:
            ik[gk_pt[0], gk_pt[1], 1] = ik_pre[gk_pre_pts[0], gk_pre_pts[1], 1]  # a channel
            ik[gk_pt[0], gk_pt[1], 2] = ik_pre[gk_pre_pts[0], gk_pre_pts[1], 2]  # b channel
          #  ik[gk_pt[0], gk_pt[1], 1] = ik_pre[gk_pt[0], gk_pt[1], 1]  # a channel
           # ik[gk_pt[0], gk_pt[1], 2] = ik_pre[gk_pt[0], gk_pt[1], 2]  # b channel

            # gk_pre_pts[0] = np.vstack((gk_pre_pts[0], cX))
            # gk_pre_pts[1] = np.vstack((gk_pre_pts[0], cY))
            # TrueIndex = []
            # FalseIndex = []
            # TrueIndex = (gk_pt[0] == gk_pre_pts[0]) & (gk_pt[1] == gk_pre_pts[1])
            # FalseIndex = (gk_pt[0] != gk_pre_pts[0]) | (gk_pt[1] != gk_pre_pts[1])
            # Value=ik[gk_pt[0][5], gk_pt[1][5], 2]
        #    ik[gk_pt[0][FalseIndex], gk_pt[1][FalseIndex], 1] = ik_pre[cX, cY, 1]
        #   ik[gk_pt[0][FalseIndex], gk_pt[1][FalseIndex], 2] = ik_pre[cX, cY, 2]
        # for p in range(len(gk_pt[0])):
        #   found = False
        #  for check in range(len(gk_pre_pts[0])):

        #     if gk_pt[0][p] == gk_pre_pts[0][check] and gk_pt[1][p] == gk_pre_pts[1][check]:
        #        ik[gk_pt[0][p], gk_pt[1][p], 1] = ik_pre[gk_pt[0][p], gk_pt[1][p], 1]
        #       ik[gk_pt[0][p], gk_pt[1][p], 2] = ik_pre[gk_pt[0][p], gk_pt[1][p], 2]
        #      found = True
        #     break
        # if not found:
        #    ik[gk_pt[p][0], gk_pt[p][1], 1] = ik_pre[cX, cY, 1]
        #   ik[gk_pt[p][0], gk_pt[p][1], 2] = ik_pre[cX, cY, 2]

        # for i in range(len(gk_pnts_list)):
        #   ik[gk_pnts_list[i][0], gk_pnts_list[i][1], 1] = ik_pre[gk_pre_pnts_list[i][0], gk_pre_pnts_list[i][1], 1]  # a channel
        #  ik[gk_pnts_list[i][0], gk_pnts_list[i][1], 2] = ik_pre[gk_pre_pnts_list[i][0], gk_pre_pnts_list[i][1], 2]  # b channel
        # ik[X, Y, 1] = ik_pre[cX, cY, 1]
        # ik[X, Y, 2] = ik_pre[cX, cY, 2]
    # Convert to RGB color model

    ik = color.lab2rgb(ik)
    ik = (ik * 255).astype("uint8")

    # Use a median filter to overcome some grayscale pixels
    # ik = cv2.bilateralFilter(ik, 100, 1, 100)
    return ik
