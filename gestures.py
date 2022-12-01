import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from tensorflow.keras.models import load_model
from pygame import mixer

# audio intialize
mixer.init()

# initialize mediapipe
mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mpDraw = mp.solutions.drawing_utils
# pose
mpPose = mp.solutions.pose
pose = mpPose.Pose()

# Load the gesture recognizer model
model = load_model('mp_hand_gesture')
# Load class names
f = open('gesture.names', 'r')
classNames = f.read().split('\n')
f.close()
print(classNames)

# Initialize the webcam for Hand Gesture Recognition Python project
cap = cv2.VideoCapture(2)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 4000)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

while True:
    # Read each frame from the webcam
    _, frame = cap.read()
    x, y, c = frame.shape

    # Flip the frame vertically
    frame = cv2.flip(frame, 1)
    framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Get hand landmark prediction
    result = hands.process(framergb)
    results = pose.process(framergb)
    className = ''

    # pose
    if results.pose_landmarks:
        landmarks = []
        mpDraw.draw_landmarks(frame, results.pose_landmarks,
                              mpPose.POSE_CONNECTIONS)
        for id, lm in enumerate(results.pose_landmarks.landmark):
            h, w, c = frame.shape
            # print(id, lm)
            cx, cy = int(lm.x * w), int(lm.y * h)
            cv2.circle(frame, (cx, cy), 5, (255, 0, 0), cv2.FILLED)

    # post process the result
    if result.multi_hand_landmarks:
        landmarks = []
        for handslms in result.multi_hand_landmarks:
            for lm in handslms.landmark:
                # print(id, lm)
                lmx = int(lm.x * x)
                lmy = int(lm.y * y)
                landmarks.append([lmx, lmy])

            # Drawing landmarks on frames
            mpDraw.draw_landmarks(frame, handslms,
                                  mpHands.HAND_CONNECTIONS)

        # Predict gesture in Hand Gesture Recognition project
        prediction = model.predict([landmarks])
        print(prediction)
        classID = np.argmax(prediction)
        className = classNames[classID]

    # audiopart
    if className == 'stop' and not mixer.music.get_busy():
        mixer.music.load("audio/text1.mp3")
        mixer.music.play()

    if className == 'okay' and not mixer.music.get_busy():
        mixer.music.load("audio/text2.mp3")
        mixer.music.play()

    if className == 'rock' and not mixer.music.get_busy():
        mixer.music.load("audio/text3.mp3")
        mixer.music.play()

    if className == 'peace' and not mixer.music.get_busy():
        mixer.music.load("audio/text4.mp3")
        mixer.music.play()

    if className == 'thumbs down' and not mixer.music.get_busy():
        mixer.music.load("audio/text5.mp3")
        mixer.music.play()

    if className == 'smile':
        mixer.music.stop()

    # show the prediction on the frame
    # cv2.putText(frame, className, (10, 50), cv2.FONT_HERSHEY_SIMPLEX,
    #             1, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, "Developed by Rishi R", org=(
        1100, 950), fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=0.5, color=(125, 246, 55), thickness=1)

    # Full screen mode
    # cv2.namedWindow('web cam', cv2.WINDOW_NORMAL)
    # cv2.setWindowProperty('web cam', cv2.WND_PROP_FULLSCREEN,
    #                       cv2.WINDOW_FULLSCREEN)

    # Show the final output
    cv2.imshow("Output", frame)
    if cv2.waitKey(1) == ord('q'):
        break

# release the webcam and destroy all active windows
cap.release()
cv2.destroyAllWindows()
