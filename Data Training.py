import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import joblib

# Step 1: Load the Data
data_folder = 'gesture_data'
data_frames = []

for gesture_file in os.listdir(data_folder):
    if gesture_file.endswith('.csv'):
        gesture_data = pd.read_csv(os.path.join(data_folder, gesture_file), header=None)
        gesture_label = gesture_file.replace('.csv', '')
        gesture_data['label'] = gesture_label
        data_frames.append(gesture_data)

# Concatenate all data frames into one
data = pd.concat(data_frames, ignore_index=True)

# Step 2: Preprocess the Data
X = data.drop('label', axis=1)  # Features (landmark coordinates)
y = data['label']  # Labels (gesture classes)

# Feature Scaling
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Label Encoding
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42)

# Step 3: Train a Machine Learning Model
classifier = SVC(kernel='linear', C=1.0, random_state=42)
classifier.fit(X_train, y_train)

# Step 4: Evaluate the Model
predictions = classifier.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print(f'Model Accuracy: {accuracy * 100:.2f}%')

# Save the trained model and scaler
joblib.dump(classifier, 'hand_gesture_model.pkl')
joblib.dump(scaler, 'scaler.pkl')

print('Model and scaler trained and saved successfully.')
