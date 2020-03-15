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
    else:
        cX, cY = 0, 0
    return [cX, cY]


def getContourPoints(img, cnt):
    """Get all the Pixels that belong to specific contour """
    mask = np.zeros(img.shape, dtype="uint8")
    mask = cv2.drawContours(mask, [cnt], -1, 255, -1)
    mask = 255 - mask
    show_images([mask])

    pnts = mask == 255
    print(pnts)
    return pnts


def ContourPropagation(gk, gk_prev, ik_pre):
    """ Propagate color through img contours"""

    ik_pre = color.rgb2lab(ik_pre)  # Convert RGB to LAB color model
    ik = np.zeros(ik_pre.shape, dtype="uint8")  # initialize the output color image
    # Convert to Binary Image
    gk = (rgb2gray(gk) * 255).astype("uint8")
    gk_prev = (gk_prev * 255).astype("uint8")
    _, gk = cv2.threshold(gk, 225, 255, cv2.THRESH_BINARY)
    _, gk_prev = cv2.threshold(gk_prev, 225, 255, cv2.THRESH_BINARY)
    show_images([gk, gk_prev])
    # get the GrayScale of current frame and reformative it in range of 0 99
    ik[:, :, 0] = (gk / 255) * 100
    # Get Image Contours
    contours_gk, hierarchy = cv2.findContours(gk, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  # for current frame
    contours_gk_pre, hierarchy = cv2.findContours(gk_prev, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  # for previous frame
    # Match Contours
    for cnt_Gk in contours_gk:
        MatchedGkPre = cnt_Gk  # finding the corresponding contour in previous frame
        for cnt_Index in range(len(contours_gk_pre)):
            ret = cv2.matchShapes(cnt_Gk, contours_gk_pre[cnt_Index], 1, 0.0)  # get the probability of matching
            if ret == 0:
                MatchedGkPre = contours_gk_pre[cnt_Index]
                contours_gk_pre.pop(cnt_Index)
                break
        # TODO: if probability is close to 1 after looping => this is a new object => ignore it
        gk_pnts = getContourPoints(gk, cnt_Gk)
        gk_pre_pnts = getContourPoints(gk_prev, MatchedGkPre)
        # copy original color value from previous
        for pnt_index in range(gk_pnts):
            if not gk_pre_pnts[pnt_index]:
                print("x")  # get pixel average
            else:
                ik[pnt_index, 1] = ik_pre[pnt_index, 1]  # a channel
                ik[pnt_index, 2] = ik_pre[pnt_index, 2]  # b channel
    # Convert to RGB color model
    ik = color.lab2rgb(ik)
    show_images([ik])
    # TODO: use median filter if somepixel are not colorized
    return ik
