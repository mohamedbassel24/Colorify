from commonfunctions import *
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
    # Back to original Size
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
        # append in the colorized list
        ColorizedFrameList.append(ContourPropagation(Gk, Gk_1, Ik_1))
        # Write the frame for debugging purposes
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
        # Insert in the colorized list from beginning
        ColorizedFrameList.insert(0, ContourPropagation(Gk, Gk_1, Ik_1))  # insert at beginning
        # Write the frame for debugging purposes
        WriteFrames(i, ColorizedFrameList[0], ShootNum + 1)
    print("[INFO] Color Propagation in shoot# ", ShootNum + 1, "is done...")
    return ColorizedFrameList


def interactiveColorization(Position, Color, Frame):
    """
    input:
            User Pixel Location [Row,Col]
            color of colorized contour [R,G,B]
            Frame to be colorized
    output: contour at this position is colorized with this color
    """

    # Convert RGB to LAB color model
    lab_Frame = color.rgb2lab(Frame)
    # Convert to Binary Image
    gk = (rgb2gray(Frame) * 255).astype("uint8")
    # GlobalThresh = 200

    # gray = cv2.bilateralFilter(gk, 11, 17, 17)
    # gk = cv2.Canny(gray, 30, 150)
    # gk = cv2.dilate(gk, None, iterations=2)
    # gk = cv2.erode(gk, None, iterations=2)

    GlobalThresh = threshold_otsu(gk)
    # Convert Image to Binary
    _, gk = cv2.threshold(gk, GlobalThresh, 255, cv2.THRESH_BINARY)
    gk = cv2.dilate(gk, None, iterations=10)
    gk = cv2.erode(gk, None, iterations=10)
    # Get Image Contours
    contours_gk, _ = cv2.findContours(gk, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  # for current frame
    show_images([gk])
    minDist = 1000  # Max Constant value
    MostMatchedContour = contours_gk[0]
    for cnt_gk in contours_gk:
        dist = cv2.pointPolygonTest(cnt_gk, Position, True)  # check if the point inside contour or not
        if minDist > dist >= 0:
            minDist = dist
            MostMatchedContour = cnt_gk

    if minDist != 1000:
        gk_pointsCnt = getContourPoints(gk, MostMatchedContour)  # points from current frame
        lab_Frame[gk_pointsCnt[0], gk_pointsCnt[1], 1] = Color[0]
        lab_Frame[gk_pointsCnt[0], gk_pointsCnt[1], 2] = Color[1]
    else:
        print("[INFO] No object Found with this location , tune pixel location")
    # Convert to RGB color model
    ik = color.lab2rgb(lab_Frame)
    ik = (ik * 255).astype("uint8")
    show_images([Frame, ik])
    return ik


def rgb2lab(inputColor):
    num = 0
    RGB = [0, 0, 0]

    for value in inputColor:
        value = float(value) / 255

        if value > 0.04045:
            value = ((value + 0.055) / 1.055) ** 2.4
        else:
            value = value / 12.92

        RGB[num] = value * 100
        num = num + 1

    XYZ = [0, 0, 0, ]

    X = RGB[0] * 0.4124 + RGB[1] * 0.3576 + RGB[2] * 0.1805
    Y = RGB[0] * 0.2126 + RGB[1] * 0.7152 + RGB[2] * 0.0722
    Z = RGB[0] * 0.0193 + RGB[1] * 0.1192 + RGB[2] * 0.9505
    XYZ[0] = round(X, 4)
    XYZ[1] = round(Y, 4)
    XYZ[2] = round(Z, 4)

    # Observer= 2°, Illuminant= D65
    XYZ[0] = float(XYZ[0]) / 95.047  # ref_X =  95.047
    XYZ[1] = float(XYZ[1]) / 100.0  # ref_Y = 100.000
    XYZ[2] = float(XYZ[2]) / 108.883  # ref_Z = 108.883

    num = 0
    for value in XYZ:

        if value > 0.008856:
            value = value ** (0.3333333333333333)
        else:
            value = (7.787 * value) + (16 / 116)

        XYZ[num] = value
        num = num + 1

    Lab = [0, 0, 0]

    L = (116 * XYZ[1]) - 16
    a = 500 * (XYZ[0] - XYZ[1])
    b = 200 * (XYZ[1] - XYZ[2])

    Lab[0] = round(L, 4)
    Lab[1] = round(a, 4)
    Lab[2] = round(b, 4)

    return Lab


def Interactive(img):
    # choice = input("[INFO] Do you want to use interactive colorization?y/n \n")
    choice = 'n'
    while choice == 'y':
        #  gk = (rgb2gray(img) * 255).astype("uint8")
        # GlobalThresh = threshold_otsu(gk)
        # Convert Image to Binary
        # _, gk = cv2.threshold(gk, GlobalThresh, 255, cv2.THRESH_BINARY)
        # gk = cv2.erode(gk, None, iterations=2)
        # gk = cv2.dilate(gk, None, iterations=2)

        # gk = cv2.erode(gk, None, iterations=10)
        # gk = cv2.dilate(gk, None, iterations=10)

        # print("Choose the White \n")
        # show_images([gk])
        row = int(input("[INFO] Enter pixel row position \n"))
        col = int(input("[INFO] Enter pixel col position \n"))
        R = int(input("[INFO] Enter color r component\n"))
        G = int(input("[INFO] Enter color g component \n"))
        B = int(input("[INFO] Enter color b component \n"))
        img = interactiveColorization((row, col), rgb2lab([R, G, B])[1:], img)
        choice = input("[INFO] Do you want to Enter another point ?y/n \n")
        if choice != 'y':
            break
    return img
