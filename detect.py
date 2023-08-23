import cv2
import numpy as np

def lane_mask(img):

    # GAUSSIAN BLUR - reduce noise
    blur = cv2.GaussianBlur(img, (5, 5), 0)

    '''
    # CONVERT TO HLS COLORSPACE - allow to filter white by "lightness"
    # You want to tune the lower and upper bound for the lighting conditions
    # the higher the number, the brigher the "lightness"
    # we want high lightness as that signifies a white color
    '''
    hls = cv2.cvtColor(blur, cv2.COLOR_BGR2HLS)
    mask = cv2.inRange(hls[:, :, 1], 220, 255)
    h, w = mask.shape

    '''
    OPENING MORPHOLOGY - reduce noise caused by white specks in pavement
    the kernel_struct size should be tuned 
        - the large the shape, the "bigger" the noise blobs it'll filter out
        - You want to find the sweet spot; 
            -if too big, it'll filter out farther away lanes that appear smaller
            - if too small, there will be excess noise
    '''
    # smaller kernel struct for lanes that are farther away (top third of image)
    kernel_struct = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    mask[0:h//3, :] = cv2.morphologyEx(mask[0:h//3, :], cv2.MORPH_OPEN, kernel_struct)
    
    # larger kernel struct for lanes that are closer (bottom third of image)
    kernel_struct = cv2.getStructuringElement(cv2.MORPH_RECT, (8, 8))
    mask[h//3:, :] = cv2.morphologyEx(mask[h//3:, :], cv2.MORPH_OPEN, kernel_struct)


    '''
    POSSIBLE FUTURE IMPROVEMENTS
    Detect lines and avoid other blobs
    - apply canny edge detector to get edges
    - do Hough line transform to get lines
    '''

    return mask

video = "comp23_2.MOV"
cap = cv2.VideoCapture(video)

while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        break

    mask = lane_mask(frame)

    frame_resize = cv2.resize(frame, (0,0), fx=0.5, fy=0.5)
    mask_resize = cv2.resize(mask, (0,0), fx=0.5, fy=0.5)
    mask_3channel = cv2.cvtColor(mask_resize, cv2.COLOR_GRAY2BGR)
    combined = np.hstack((frame_resize, mask_3channel))

    cv2.imshow("vid", combined)

    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
