import cv2
from tensorflow.keras.models import model_from_json
from datagenerator import FacialKeyPointsDataset
import numpy as np

TYPE_OF_DATA_AND_MODEL = 'vector'

face_cascade = cv2.CascadeClassifier(
    'detector_architectures/haarcascade_frontalface_default.xml')

datasetgen = FacialKeyPointsDataset(csv_file='data/training_frames_keypoints.csv',
                                    root_dir='data/training/',
                                    normalization=TYPE_OF_DATA_AND_MODEL)

# load json and create model
json_file = open(
    'models/model_{}_batchnorm_194.json'.format(TYPE_OF_DATA_AND_MODEL),
    'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
# load weights into new model
model.load_weights(
    "models/model_{}_batchnorm_194.h5".format(TYPE_OF_DATA_AND_MODEL))
print("Loaded model from disk")


def detectFace(image):
    # run the detector
    faces = face_cascade.detectMultiScale(image, 1.2, 10, 30)

    # make a copy of the original image to plot detections on
    image_with_detections = image.copy()

    # loop over the detected faces, mark the image where each face is found
    for (x, y, w, h) in faces:
        # draw a rectangle around each detected face
        cv2.putText(image_with_detections, 'Hooman', (x, y-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)
        cv2.rectangle(image_with_detections, (x, y),
                      (x + w, y + h), (255, 0, 0), 3)

    return image_with_detections, faces


def get_key_points(frame, face):
    pad = 50
    (x, y, w, h) = face
    roi = frame[y - pad:y + h + pad, x - pad:x + w + pad]
    originalSize = roi.shape
    if(originalSize[0] == 0 or originalSize[1] == 0):
        return (), roi, originalSize, (0, 0, 0, 0)

    roi = cv2.resize(roi, datasetgen.output_size)
    grayscale = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
    img = grayscale/255.
    img = img.reshape(1, *grayscale.shape, 1)
    keypts = model.predict(img).reshape(-1, 1)
    keypts = keypts * datasetgen.std + datasetgen.mean
    keypts = keypts.reshape(-1, 2)

    return keypts, roi, originalSize, (x-pad, y-pad, w+pad*2, h+pad*2)


cap = cv2.VideoCapture(0)

while True:
    keyPressed = cv2.waitKey(1) & 0xFF
    if keyPressed == ord('q'):
        break

    ret, frame = cap.read()
    if(frame.shape[0] == 0 or frame.shape[1] == 0):
        continue
    frame = cv2.resize(frame, (720, 480))
    frame, faces = detectFace(frame)

    if faces is not None:
        for face in faces:
            keypts, roi, originalSize, (x, y, w, h) = get_key_points(
                frame, face)
            for pt in keypts:
                img_with_pts = cv2.circle(roi, tuple(
                    pt.astype(np.int)), 1, (100, 200, 0), -1)
            if (img_with_pts.shape[0] == 0 or img_with_pts.shape[1] == 0):
                continue
            # print(tuple(originalSize[:2]))
            img_with_pts = cv2.resize(
                img_with_pts,
                (originalSize[1], originalSize[0])
            )
            frame[y:y+h, x:x+w] = img_with_pts
    cv2.imshow("Frame", frame)

cap.release()
