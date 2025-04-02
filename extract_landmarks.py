import cv2
import mediapipe as mp
import os
import csv
import numpy as np

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)

# Dataset path
train_dir = "asl_alphabet_train/asl_alphabet_train"

# Output CSV
output_file = "asl_landmarks.csv"

# Headers for CSV (label + 63 features for one hand)
headers = ["label"] + [f"hand_{i}_{coord}" for i in range(21) for coord in ("x", "y", "z")]

# Open CSV file
with open(output_file, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(headers)

# Process each class folder
for class_name in os.listdir(train_dir):
    class_path = os.path.join(train_dir, class_name)
    if not os.path.isdir(class_path):
        continue
    
    print(f"Processing class: {class_name}")
    for img_name in os.listdir(class_path):
        img_path = os.path.join(class_path, img_name)
        img = cv2.imread(img_path)
        if img is None:
            continue
        
        # Convert to RGB for MediaPipe
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Extract landmarks (x, y, z for 21 points)
                landmarks = [coord for landmark in hand_landmarks.landmark for coord in (landmark.x, landmark.y, landmark.z)]
                with open(output_file, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([class_name] + landmarks)
        else:
            # If no hand detected, skip or log (optional)
            print(f"No hand detected in {img_path}")

hands.close()
print("Landmark extraction complete. Saved to 'asl_landmarks.csv'")