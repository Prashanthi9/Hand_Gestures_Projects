import cv2
import mediapipe as mp

# Initialize MediaPipe Hands.
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils

# Start video capture from the default webcam.
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            landmark_list = []

            # Get landmark positions.
            for id, lm in enumerate(hand_landmarks.landmark):
                h, w, c = frame.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                landmark_list.append([cx, cy])

            # Check for gestures if landmarks are available.
            if len(landmark_list) != 0:
                gesture = None
                # Gesture: Syntax Sarcasm (Thumb and Index finger raised)
                if landmark_list[4][1] < landmark_list[3][1] and landmark_list[8][1] < landmark_list[6][1]:
                    gesture = "Syntax Sarcasm"
                # Gesture: Join the Workshop (Thumb down, Index finger raised)
                elif landmark_list[4][1] > landmark_list[3][1] and landmark_list[8][1] < landmark_list[6][1]:
                    gesture = "Join the workshop"

                # Display the gesture on the frame.
                if gesture:
                    cv2.putText(frame, gesture, (landmark_list[0][0] - 50, landmark_list[0][1] - 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3, cv2.LINE_AA)

    # Display the frame.
    cv2.imshow('Hand Gesture Recognition', frame)

    # Break the loop if 'q' is pressed.
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close all windows.
cap.release()
cv2.destroyAllWindows()
