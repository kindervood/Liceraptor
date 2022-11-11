# определяет кто стоит перед камерой из команды

import imutils
from imutils.video import FPS
import face_recognition
import pickle
import time
import cv2
from deepface import DeepFace

# Имя для неопознанного
currentname = "unknown"
# модель с тренировки
encodingsP = r"encodings.pickle"

print("[INFO] loading encodings + face detector...")
data = pickle.loads(open(encodingsP, "rb").read())


cam = cv2.VideoCapture(0)
# vs = VideoStream(usePiCamera=True).start()




backends = ['opencv', 'ssd', 'dlib', 'mtcnn', 'retinaface',
            'mediapipe']  # opencv или ssd - скорость \ retinaface и mtcnn - качество


def face_analyze(img1):
    try:
        res_attributes = DeepFace.analyze(img_path=img1, actions=("age", "emotion"), detector_backend=backends[0])

        return res_attributes

    except Exception as _ex:
        return _ex


fps = FPS().start()

# видеопотое
while True:
    ret, frame = cam.read()
    frame = imutils.resize(image=frame, width=1000)
    boxes = face_recognition.face_locations(frame)
    encodings = face_recognition.face_encodings(frame, boxes)
    names = []

    for encoding in encodings:
        matches = face_recognition.compare_faces(data["encodings"],
                                                 encoding)

        name = "Who are you, dino -_-"  # если не совпало

        # попало ли совпадение
        if True in matches:
            # слоаврь для подсчета общего кол-ва раз совпадающего лица
            matchedIdxs = [i for (i, b) in enumerate(matches) if b]
            counts = {}

            for i in matchedIdxs:
                name = data["names"][i]
                counts[name] = counts.get(name, 0) + 1

            name = max(counts, key=counts.get)

            # res = face_analyze(frame)


            if currentname != name:
                currentname = name
                print(currentname)


        names.append(name)

    # перебор распознанных лиц
    for ((top, right, bottom, left), name) in zip(boxes, names):
        cv2.rectangle(frame, (left, top), (right, bottom),
                      (0, 255, 225), 2)
        y = top - 15 if top - 15 > 15 else top + 15
        cv2.putText(frame, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX,
                    .8, (0, 255, 255), 2)

    # display the image to our screen
    cv2.imshow("Facial Recognition is Running", frame)
    key = cv2.waitKey(1) & 0xFF

    # quit when 'q' key is pressed
    if key == ord("q"):
        break
    elif key == ord(" "):
        res = face_analyze(frame)
        if str(type(res)) == "<class 'dict'>":
            res1 = res["dominant_emotion"]
            res2 = res["age"]

            print(res1, res2)
            with open("base.txt", "w") as file:
                file.write(currentname + "\t" + res1 + "\t" + str(res2))
            time.sleep(3)

    # update the FPS counter
    fps.update()

# stop the timer and display FPS information
fps.stop()
print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

# do a bit of cleanup
cv2.destroyAllWindows()
