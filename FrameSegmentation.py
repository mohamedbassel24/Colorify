# Program To Read video
# and Extract Frames
import cv2


# Function to extract frames
def FrameCapture(path):
    # Path to video file
    vidObj = cv2.VideoCapture(path)

    # Used as counter variable
    count = 0

    # checks whether frames were extracted
    success = 1

    while success:
        # vidObj object calls read
        # function extract frames
        success, image = vidObj.read()

        #comment this line for colorized images
        image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

        # Saves the frames with frame-count
        cv2.imwrite(r"Test Video\\Segmented Frames GRAYSCALE Balloons\\frame%d.jpg" % count, image)
    #### WILL GIVE UNKNOWN ERROR BUT OUTPUTS CORRECTLY DONT WASTE UR TIME
        count += 1


# Driver Code
if __name__ == '__main__':
    # Calling the function
    FrameCapture("C:\\Users\\Mohamed Haitham\\Documents\\GitHub\\Color-IT\\Test Video\\balloons.mp4")
