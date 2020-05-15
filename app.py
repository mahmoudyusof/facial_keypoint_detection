import cv2
from tensorflow.keras.models import model_from_json
from datagenerator import FacialKeyPointsDataset
import numpy as np

TYPE_OF_DATA_AND_MODEL = 'vector'

flags = {
    "detect_faces": False,
    "draw_keypts": False,
    "filter": False,
    "run": True
}

filters = {
    "filter1": {
        "img": "images/filter1.png",
        "coord": 17,
        "h": (25, 29),
        "w": (16, 2),
        "padding": (0, 0),
        "offset": (-10, 0)
    },
    "filter2": {
        "img": "images/filter2.png",
        "coord": 17,
        "h": (20, 9),
        "w": (2, 15),
        "padding": (30, 40),
        "offset": (-20, -70)
    }
}

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


def draw_key_pts(frame, face, keypts, rof, cosize):
    x, y, w, h = cosize
    for pt in keypts:
        rof = cv2.circle(rof, tuple(
            pt.astype(np.int)), 1, (100, 200, 0), -1)

    if (rof.shape[0] == 0 or rof.shape[1] == 0):
        return frame

    rof = cv2.resize(
        rof,
        (originalSize[1], originalSize[0])
    )
    frame[y:y+h, x:x+w] = rof
    return frame


def add_filter(frame, filter, face, keypts, rof, cosize):
    path = filters[filter]['img']
    filter_img = cv2.imread(path, -1)

    w = int(abs(
        keypts[filters[filter]['w'][0]][0] - keypts[filters[filter]['w'][1]][0]
    )) + filters[filter]['padding'][0]

    h = int(abs(
        keypts[filters[filter]['h'][0]][1] - keypts[filters[filter]['h'][1]][1]
    )) + filters[filter]['padding'][1]

    filter_img = cv2.resize(filter_img, (w, h), interpolation=cv2.INTER_CUBIC)

    x = int(keypts[filters[filter]['coord']][0]) + filters[filter]['offset'][0]
    y = int(keypts[filters[filter]['coord']][1]) + filters[filter]['offset'][1]

    roi_color = rof[y:y + h, x:x + w]

    non_trans = np.argwhere(filter_img[:, :, 3] > 0)

    if(non_trans.shape[0] > roi_color.shape[0]*roi_color.shape[1]):
        return frame

    roi_color[non_trans[:, 0], non_trans[:, 1],
              :3] = filter_img[non_trans[:, 0], non_trans[:, 1], :3]
    rof[y:y + h, x:x + w] = roi_color

    rof = cv2.resize(rof, (originalSize[1], originalSize[0]))

    x2, y2, w2, h2 = cosize

    frame[y2:y2 + h2, x2:x2 + w2] = rof

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
        flags['filter'] = "filter1" if flags['filter'] != "filter1" else False
    elif key == ord('4'):
        flags['filter'] = "filter2" if flags['filter'] != "filter2" else False
    elif key == ord('q'):
        flags['run'] = False


if __name__ == '__main__':
    cap = cv2.VideoCapture(0)
    while flags['run']:
        keyPressed = cv2.waitKey(1) & 0xFF

        handle_key_press(keyPressed)

        ret, frame = cap.read()
        if(frame.shape[0] == 0 or frame.shape[1] == 0):
            continue

        frame = cv2.resize(frame, (720, 480))
        frame, faces = detectFace(frame)

        if faces is not None and (flags['draw_keypts'] or flags['filter']):
            for face in faces:
                keypts, rof, originalSize, cosize = get_key_points(
                    frame, face)
                if flags['draw_keypts']:
                    frame = draw_key_pts(frame, face, keypts, rof, cosize)
                if flags['filter']:
                    frame = add_filter(
                        frame, flags['filter'], face, keypts, rof, cosize)

        cv2.imshow("Frame", frame)
    cap.release()
