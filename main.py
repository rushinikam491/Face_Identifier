import cv2

faceProto = "opencv_face_detector.pbtxt"
faceModel = "opencv_face_detector_uint8.pb"

ageProto = "age_deploy.prototxt"
ageModel = "age_net.caffemodel"

genderProto = "gender_deploy.prototxt"
genderModel = "gender_net.caffemodel"

facenet = cv2.dnn.readNet(faceModel, faceProto)
agenet = cv2.dnn.readNet(ageModel, ageProto)
gendernet = cv2.dnn.readNet(genderModel, genderProto)

ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
genderList = ['Male', 'Female']

face_cap = cv2.CascadeClassifier("C:/Users/rushi/AppData/Local/Programs/Python/Python311/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml")
smileCascade = cv2.CascadeClassifier('haarcascade_smile.xml')
video_cap = cv2.VideoCapture(0)

while True:
    ret, video = video_cap.read()
    gray = cv2.cvtColor(video, cv2.COLOR_BGR2GRAY)
    faces = face_cap.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30),
                                      flags=cv2.CASCADE_SCALE_IMAGE)
    for (x, y, w, h) in faces:
        cv2.rectangle(video, (x, y), (x + w, y + h), (255, 255, 255), 3)
        face_roi = gray[y:y + h, x:x + w]
        face_roi_color = cv2.cvtColor(face_roi, cv2.COLOR_GRAY2BGR)
        blob = cv2.dnn.blobFromImage(face_roi_color, 1.0, (227, 227), (78.4263377603, 87.7689143744, 114.895847746),
                                     swapRB=False)

        gendernet.setInput(blob)
        gender_preds = gendernet.forward()
        gender = genderList[gender_preds[0].argmax()]

        agenet.setInput(blob)
        age_preds = agenet.forward()
        age = ageList[age_preds[0].argmax()]

        cv2.putText(video, f"Age: {age}", (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(video, f"Gender: {gender}", (x, y - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2, cv2.LINE_AA)

        smile = smileCascade.detectMultiScale(face_roi, scaleFactor=1.5, minNeighbors=15, minSize=(25, 25))

        for i in smile:
            cv2.putText(video, "Smile Detected", (x, y - 100), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0, 0, 0), 2, cv2.LINE_AA)

    cv2.imshow("video_live", video)

    if cv2.waitKey(10) == ord("s"):
        break

video_cap.release()
cv2.destroyAllWindows()
