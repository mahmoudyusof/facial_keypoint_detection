import cv2

face_cascade = cv2.CascadeClassifier(
    'detector_architectures/haarcascade_frontalface_default.xml')


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
    cv2.imshow("Frame", frame)

cap.release()
