from commonfunctions import *


def LoadColorizationModel():
    # load our serialized black and white colorizer model and cluster
    # center points from disk
    print("[INFO] loading model...")
    net = cv2.dnn.readNetFromCaffe("model/colorization_deploy_v2.prototxt", "model/colorization_release_v2.caffemodel")
    pts = np.load("model/pts_in_hull.npy")
    # add the cluster centers as 1x1 convolutions to the model
    class8 = net.getLayerId("class8_ab")
    conv8 = net.getLayerId("conv8_313_rh")
    pts = pts.transpose().reshape(2, 313, 1, 1)
    net.getLayer(class8).blobs = [pts.astype("float32")]
    net.getLayer(conv8).blobs = [np.full([1, 313], 2.606, dtype="float32")]
    return net


def colorization(frame, net):
    """Take input image and colorize it based on given model"""

    # resize the input frame, scale the pixel intensities to the
    # range [0, 1], and then convert the frame from the BGR to Lab
    # color space
    # frame = imutils.resize(frame, width=args["width"])
    scaled = frame.astype("float32") / 255.0
    lab = cv2.cvtColor(scaled, cv2.COLOR_BGR2LAB)

    # resize the Lab frame to 224x224 (the dimensions the colorization
    # network accepts), split channels, extract the 'L' channel, and
    # then perform mean centering
    resized = cv2.resize(lab, (224, 224))
    L = cv2.split(resized)[0]
    L -= 50

    # pass the L channel through the network which will *predict* the
    # 'a' and 'b' channel values
    net.setInput(cv2.dnn.blobFromImage(L))
    ab = net.forward()[0, :, :, :].transpose((1, 2, 0))

    # resize the predicted 'ab' volume to the same dimensions as our
    # input frame, then grab the 'L' channel from the *original* input
    # frame (not the resized one) and concatenate the original 'L'
    # channel with the predicted 'ab' channels
    ab = cv2.resize(ab, (frame.shape[1], frame.shape[0]))
    L = cv2.split(lab)[0]
    colorized = np.concatenate((L[:, :, np.newaxis], ab), axis=2)

    # convert the output frame from the Lab color space to RGB, clip
    # any values that fall outside the range [0, 1], and then convert
    # to an 8-bit unsigned integer ([0, 255] range)
    colorized = cv2.cvtColor(colorized, cv2.COLOR_LAB2BGR)
    colorized = np.clip(colorized, 0, 1)
    colorized = (255 * colorized).astype("uint8")
    colorized = cv2.cvtColor(colorized, cv2.COLOR_BGR2RGB)
    return colorized
