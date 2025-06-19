import cv2 as cv
import mediapipe as mp

mp_face = mp.solutions.face_detection
mp_draw = mp.solutions.drawing_utils

face_detection = mp_face.FaceDetection(model_selection=0, min_detection_confidence=0.6)

cap = cv.VideoCapture(0)

# Landmark feature names (from MediaPipe order)
landmark_names = [
    "REY",
    "LEY",
    "NT",
    "MC",
    "RER",
    "lER"
]

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    result = face_detection.process(rgb_frame)

    if result.detections:
        for detection in result.detections:
            mp_draw.draw_detection(frame, detection)

            # Extract landmark points
            h, w, _ = frame.shape
            keypoints = detection.location_data.relative_keypoints

            for i, kp in enumerate(keypoints):
                x = int(kp.x * w)
                y = int(kp.y * h)
                cv.circle(frame, (x, y), 4, (0, 0, 255), -1)
                cv.putText(frame, landmark_names[i], (x + 5, y - 5),
                           cv.FONT_HERSHEY_SIMPLEX, 0.5, (0,255, 0), 1)

    cv.imshow("Face Detection with Feature Names", frame)

    if cv.waitKey(1) == 27:  # ESC
        break

cap.release()
cv.destroyAllWindows()
