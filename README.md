# SSVIP-Artificial_Intelligence-Task_2


Data Training:
- Combined gesture data from multiple CSV files, creating a comprehensive dataset.
- Extracted features (landmark coordinates) and corresponding labels (gesture names).
- Processed and prepared the data for machine learning using scaling and encoding techniques.
- Split the data into training and testing sets, reserving a portion for model evaluation.
- Trained a Support Vector Machine (SVM) model to recognize hand gestures based on the prepared data.

Hand Gesture Recognition:
- Captured continuous video frames, ensuring real-time analysis.
- Prepared frames for analysis, flipping them for better visualization and converting them to a compatible format.
- Utilized the Mediapipe Hands module to detect and analyze hand landmarks in the frames.
- Processed landmark coordinates and used the trained SVM model to predict the corresponding gesture.
- Displayed the recognized gesture on the screen and allowed users to interact by pressing the 'q' key to exit the application.
