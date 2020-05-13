import numpy as np
import os
import cv2
from tensorflow.keras.models import model_from_json
from datagenerator import FacialKeyPointsDataset

datasetgen = FacialKeyPointsDataset(csv_file='data/training_frames_keypoints.csv',
                                             root_dir='data/training/',
                                             output_size=(194, 194),
                                             batch_size=30,
                                             normalization="vector")

# load json and create model
json_file = open('models/model_vector_batchnorm_194.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
# load weights into new model
model.load_weights("models/model_vector_batchnorm_194.h5")
print("Loaded model from disk")


def detectFace(image):
    # load in a haar cascade classifier for detecting frontal faces
    face_cascade = cv2.CascadeClassifier(
        'detector_architectures/haarcascade_frontalface_default.xml')

    # run the detector
    faces = face_cascade.detectMultiScale(image, 1.2, 10, 30)

    # make a copy of the original image to plot detections on
    image_with_detections = image.copy()

    # loop over the detected faces, mark the image where each face is found
    for (x, y, w, h) in faces:
        # draw a rectangle around each detected face
        cv2.putText(image_with_detections, 'Face Detected', (x, y-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)
        cv2.rectangle(image_with_detections, (x, y),
                      (x + w, y + h), (255, 0, 0), 3)

    return image_with_detections, faces


def detectKeys(image, faces):
    image_copy = np.copy(image)
    padding = 50
    predicted_key_pts = ()
    roi = ()
    originalSize = ()
    (x, y, w, h) = (0, 0, 0, 0)

    # loop over the detected faces from haar cascade
    for (x, y, w, h) in faces:
        # Select the region of interest that is the face in the image
        roi = image_copy[y - padding:y + h +
                         padding, x - padding:x + w + padding]
        originalSize = roi.shape

        if(roi.shape[0] == 0 or roi.shape[1] == 0):
            return predicted_key_pts, roi, originalSize, (x, y, w, h)

        roi = cv2.resize(roi, (194, 194))
        grayscale = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
        grayscale = grayscale / 255.0

        img = grayscale.reshape(1, grayscale.shape[0], grayscale.shape[1], 1)

        output_pts = model.predict(img).reshape(-1, 1)
        # undo normalization of keypoints
        output_pts = output_pts * datasetgen.std + datasetgen.mean
        output_pts = output_pts.reshape(-1, 2)

    return output_pts, roi, originalSize, (x - padding, y - padding, w + padding*2, h + padding*2)


# USE KEYBOARD NUMBERS BUTTONS TO CHANGE FILTERS
# 1 -> NORMAL CAMERA
# 2 -> DETECTED FACE
# 3 -> DETECTED KEYPOINTS
# 4 -> FILTER 1
# 5 -> FILTER 2
# q -> QUIT
def checkKeyPressed(keyPressed):
    global bface
    global bKey
    global bfilter1
    global bfilter2
    global bfilter3

    if keyPressed == ord('1'):
        print("1")
        bface = 0
        bKey = 0

    if keyPressed == ord('2'):
        print("2")
        bface = 1
        bKey = 0

    if keyPressed == ord('3'):
        print("3")
        bface = 1
        bKey = 1
        bfilter1 = 0
        bfilter2 = 0
        bfilter3 = 0

    if keyPressed == ord('4'):
        print('4')
        bfilter1 = 1
        bfilter2 = 0
        bfilter3 = 0
        bface = 1
        bKey = 1

    if keyPressed == ord('5'):
        print('5')
        bfilter1 = 0
        bfilter2 = 1
        bfilter3 = 0
        bface = 1
        bKey = 1

    if keyPressed == '6':
        bfilter1 = 0
        bfilter2 = 0
        bfilter3 = 1
        bface = 1
        bKey = 1


def addFilter(filter, roi, originalSize, image, x, y, w, h, x2, y2, w2, h2):
    # resize filter
    filter = cv2.resize(filter, (w, h), interpolation=cv2.INTER_CUBIC)

    # get region of interest on the face to change
    roi_color = roi[y:y + h, x:x + w]

    # find all non-transparent pts
    ind = np.argwhere(filter[:, :, 3] > 0)
    # for each non-transparent point, replace the original image pixel with that of the new_sunglasses
    if(ind.shape[0] > roi_color.shape[0]*roi_color.shape[1]):
        return image

    for i in range(3):
        roi_color[ind[:, 0], ind[:, 1], i] = filter[ind[:, 0], ind[:, 1], i]
        # set the area of the image to the changed region with sunglasses
    roi[y:y + h, x:x + w] = roi_color

    roi = cv2.resize(roi, (originalSize[1], originalSize[0]))
    image[y2:y2 + h2, x2:x2 + w2] = roi

    return image


cap = cv2.VideoCapture(0)

bface = False
bKey = False
bfilter1 = False
bfilter2 = False
bfilter3 = False

filter1 = cv2.imread('images/filter1.png', -1)
filter2 = cv2.imread('images/filter2.png', -1)

while(True):
    ret, frame = cap.read()
    frame = cv2.resize(frame, (720, 480))
    cv2.putText(frame,
                'Press Buttons 1:Regular Camera     2:Detected Face    3:Detected Keypoints    4:Filter1     5:Filter2   q:Quit',
                (10, 480-20), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA)

    keyPressed = cv2.waitKey(1) & 0xFF

    if(frame.shape[0] == 0 or frame.shape[1] == 0):
        continue

    checkKeyPressed(keyPressed)

    if bface:
        detectedFacesImage, faces = detectFace(frame)
        if bKey and faces is not None:
            predicted_key_pts, roi, originalSize, (x, y, w, h) = detectKeys(
                frame, faces)
            if(not(bfilter1 or bfilter2 or bfilter3) and predicted_key_pts.size > 0):
                for key in predicted_key_pts:
                    detectedKeysImage = cv2.circle(
                        roi, tuple(key.astype(np.int)), 1, (100, 200, 0), -1)

                if (detectedKeysImage.shape[0] == 0 or detectedKeysImage.shape[1] == 0):
                    continue
                detectedKeysImage = cv2.resize(
                    detectedKeysImage, (originalSize[1], originalSize[0]))
                frame[y:y + h, x:x + w] = detectedKeysImage
    if(bfilter1 and predicted_key_pts.size > 0 and roi.shape[0] != 0 and roi.shape[1] != 0):
        finalImage = addFilter(filter1, roi, originalSize, frame.copy(),
                               x=int(predicted_key_pts[17][0] - 10),
                               y=int(predicted_key_pts[17][1]),
                               h=int(
                                   abs(predicted_key_pts[25][1] - predicted_key_pts[29][1])),
                               w=int(abs(predicted_key_pts[16][0] - predicted_key_pts[2][0])), x2=x, y2=y, w2=w, h2=h
                               )
        cv2.imshow("Frame", finalImage)
    elif (bfilter2 and predicted_key_pts.size > 0 and roi.shape[0] != 0 and roi.shape[1] != 0):
        finalImage = addFilter(filter2, roi, originalSize, frame.copy(),
                               x=int(predicted_key_pts[17][0] - 20),
                               y=int(predicted_key_pts[17][1] - 70),
                               h=int(
                                   abs(predicted_key_pts[20][1] - predicted_key_pts[9][1]) + 40),
                               w=int(abs(predicted_key_pts[2][0] - predicted_key_pts[15][0]) + 30), x2=x, y2=y, w2=w,
                               h2=h
                               )
        cv2.imshow("Frame", finalImage)
    elif(bKey):
        cv2.imshow("Frame", frame)
    elif(bface):
        cv2.imshow("Frame", detectedFacesImage)
    else:
        cv2.imshow("Frame", frame)

    if keyPressed == ord('q'):
        break
