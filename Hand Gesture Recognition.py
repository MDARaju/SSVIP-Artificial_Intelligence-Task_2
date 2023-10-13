import cv2
import mediapipe as mp
import joblib
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Initialize Mediapipe Hands module
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

# Load the trained model and scaler
classifier = joblib.load('hand_gesture_model.pkl')
scaler = joblib.load('scaler.pkl')

# Gesture labels mapping
gesture_labels = {
    1: "Fist",
    2: "High Five",
    3: "Peace",
    4: "Thumbs Down",
    5: "Thumbs Up",
}

# Initialize video capture
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    # Flip the frame horizontally for a later selfie-view display
    frame = cv2.flip(frame, 1)

    # Convert the frame to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame with Mediapipe Hands
    results = hands.process(frame_rgb)

    # Draw landmarks on the frame and predict the gesture
    if results.multi_hand_landmarks:
        for landmarks in results.multi_hand_landmarks:
            landmark_coords = []
            for landmark in landmarks.landmark:
                landmark_x = landmark.x
                landmark_y = landmark.y
                landmark_coords.extend([landmark_x, landmark_y])

            # Preprocess the data and make predictions
            landmark_coords = np.array(landmark_coords).reshape(1, -1)
            landmark_coords_scaled = scaler.transform(landmark_coords)  # Scale features
            predicted_label_encoded = classifier.predict(landmark_coords_scaled)[0]

            # Get the meaningful gesture label
            predicted_label = gesture_labels.get(predicted_label_encoded + 1 , "Unknown")

            # Display the recognized gesture on the frame
            cv2.putText(frame, predicted_label, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow('Hand Gesture Recognition', frame)

    # Break the loop and stop capturing data when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture
cap.release()
cv2.destroyAllWindows()
