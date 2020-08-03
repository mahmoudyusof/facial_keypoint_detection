import cv2
from tensorflow.keras.models import model_from_json
import numpy as np
from datagenerator import FacialKeyPointsDataset
from filters import Filter

TYPE_OF_DATA_AND_MODEL = 'vector'

flags = {
    "detect_faces": False,
    "draw_keypts": False,
    "filter": None,
    "run": True
}

filters = [
    Filter("images/filter1.png", 17, (25, 29), (2, 16), offset=(-10, 0)),
]

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

    if flags['detect_faces']:
        # loop over the detected faces, mark the image where each face is found
        for (x, y, w, h) in faces:
            # draw a rectangle around each detected face
            cv2.putText(image_with_detections, 'Hooman', (x, y-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)
            cv2.rectangle(image_with_detections, (x, y),
                          (x + w, y + h), (255, 0, 0), 3)

    return image_with_detections, faces


def get_key_points(frame, face):
    # padding to get the whole face
    pad = 50
    (x, y, w, h) = face
    # get the region of interest from the frame
    roi = frame[y - pad:y + h + pad, x - pad:x + w + pad]
    # we'll need that later
    originalSize = roi.shape

    if(originalSize[0] == 0 or originalSize[1] == 0):
        return (), roi, originalSize, (0, 0, 0, 0)

    resized_roi = cv2.resize(roi, datasetgen.output_size)
    # preprocess for prediction
    img = datasetgen.preprocess_test(roi)
    # make a batch
    img = img.reshape(1, *img.shape, 1)
    # predict
    keypts = model.predict(img).reshape(-1, 1)
    # undo normalization of keypoints
    keypts = keypts * datasetgen.std + datasetgen.mean
    # make (x, y) coordinates from points vector
    keypts = keypts.reshape(-1, 2)

    return keypts, resized_roi, originalSize, (x-pad, y-pad, w+pad*2, h+pad*2)


def draw_key_pts(frame, face, keypts, roi, cosize):
    # the size of the image on the frame
    x, y, w, h = cosize
    # draw a small point for each keypoint on the region of face
    for pt in keypts:
        roi = cv2.circle(roi, tuple(
            pt.astype(np.int)), 1, (100, 200, 0), -1)

    if (roi.shape[0] == 0 or roi.shape[1] == 0):
        return frame

    # resizef for numpy use
    roi = cv2.resize(
        roi,
        (originalSize[1], originalSize[0])
    )
    # replace pixle data
    frame[y:y+h, x:x+w] = roi
    return frame


def handle_key_press(key):
    global flags
    if key == ord('1'):
        flags['detect_faces'] = not flags['detect_faces']
        print(flags['detect_faces'])
    elif key == ord('2'):
        flags['draw_keypts'] = not flags['draw_keypts']
        print(flags['draw_keypts'])
    elif key == ord('3'):
        flags['filter'] = None if flags['filter'] is not None else 0
    elif key == ord('q'):
        flags['run'] = False


if __name__ == '__main__':
    # capture video
    cap = cv2.VideoCapture(0)
    while flags['run']:
        # get any keypress
        keyPressed = cv2.waitKey(1) & 0xFF
        handle_key_press(keyPressed)
        # get the frame
        ret, frame = cap.read()
        if(frame.shape[0] == 0 or frame.shape[1] == 0):
            continue
        # resize the frame and detect faces
        frame = cv2.resize(frame, (720, 480))
        cv2.putText(frame,
                    'Toggle Buttons - 1:Detected Face    2:Detected Keypoints    3:Filter1    q:Quit',
                    (10, 480-20), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA)
        frame, faces = detectFace(frame)

        if faces is not None and (flags['draw_keypts'] or flags['filter'] is not None):
            for face in faces:
                # get the keypoints, you will need them the both of the cases
                keypts, roi, originalSize, cosize = get_key_points(
                    frame, face)
                if flags['draw_keypts']:
                    frame = draw_key_pts(frame, face, keypts, roi, cosize)
                if flags['filter'] is not None:
                    # cosize is the size to be sliced off of the frame
                    x, y, w, h = cosize
                    # apply the filter
                    roi = filters[flags['filter']].apply(keypts, roi)
                    # get the roi to the original size and ready to be sliced back
                    roi = cv2.resize(roi, (originalSize[1], originalSize[0]))
                    # slice it back
                    frame[y:y+h, x:x+w] = roi

        cv2.imshow("Frame", frame)
    cap.release()
