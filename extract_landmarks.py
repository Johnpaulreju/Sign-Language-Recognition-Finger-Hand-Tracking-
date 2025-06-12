# import cv2
# import mediapipe as mp
# import os
# import csv
# import numpy as np
# import shutil

# # Initialize MediaPipe Hands
# mp_hands = mp.solutions.hands
# hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.3)

# # Dataset path
# train_dir = "asl_alphabet_train/asl_alphabet_train"

# # Output CSV
# output_file = "asl_landmarks.csv"

# # Directory to save problematic images
# failed_dir = "failed_images"
# os.makedirs(failed_dir, exist_ok=True)

# # Headers for CSV (label + 63 features for one hand)
# headers = ["label"] + [f"hand_{i}_{coord}" for i in range(21) for coord in ("x", "y", "z")]

# # Open CSV file
# with open(output_file, "w", newline="") as f:
#     writer = csv.writer(f)
#     writer.writerow(headers)

# # Get list of class folders and sort them alphabetically
# class_folders = sorted([d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))])

# # Process each class folder in alphabetical order
# for class_name in class_folders:
#     class_path = os.path.join(train_dir, class_name)
    
#     print(f"Processing class: {class_name}")
#     # Sort images in the folder alphabetically
#     img_list = sorted(os.listdir(class_path))
#     for img_name in img_list[:500]:  # Limit to 500 images per class to speed up
#         img_path = os.path.join(class_path, img_name)
#         img = cv2.imread(img_path)
#         if img is None:
#             print(f"Failed to load image: {img_path}")
#             continue
        
#         # Convert to RGB for MediaPipe
#         img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#         results = hands.process(img_rgb)

#         if results.multi_hand_landmarks:
#             for hand_landmarks in results.multi_hand_landmarks:
#                 # Extract landmarks (x, y, z for 21 points)
#                 landmarks = [coord for landmark in hand_landmarks.landmark for coord in (landmark.x, landmark.y, landmark.z)]
#                 with open(output_file, "a", newline="") as f:
#                     writer = csv.writer(f)
#                     writer.writerow([class_name] + landmarks)
#         else:
#             # Log and save the problematic image
#             print(f"No hand detected in {img_path}")
#             shutil.copy(img_path, os.path.join(failed_dir, f"{class_name}_{img_name}"))

# hands.close()
# print("Landmark extraction complete. Saved to 'asl_landmarks.csv'")
# print(f"Problematic images saved to '{failed_dir}'")













import cv2
import mediapipe as mp
import os
import csv
import numpy as np
import shutil

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.1)

# Dataset path
train_dir = "asl_alphabet_train/asl_alphabet_train"

# Output CSV
output_file = "asl_landmarks.csv"

# Directory to save problematic images
failed_dir = "failed_images"
os.makedirs(failed_dir, exist_ok=True)

# Headers for CSV (label + 63 features for one hand)
headers = ["label"] + [f"hand_{i}_{coord}" for i in range(21) for coord in ("x", "y", "z")]

# Open CSV file
with open(output_file, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(headers)

# Get list of class folders and sort them alphabetically
class_folders = sorted([d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))])

# Process each class folder in alphabetical order
for class_name in class_folders:
    class_path = os.path.join(train_dir, class_name)
    
    print(f"Processing class: {class_name}")
    # Sort images in the folder alphabetically
    img_list = sorted(os.listdir(class_path))
    success_count = 0
    for img_name in img_list[:500]:  # Limit to 500 images per class
        img_path = os.path.join(class_path, img_name)
        img = cv2.imread(img_path)
        if img is None:
            print(f"Failed to load image: {img_path}")
            continue
        
        # Resize image to 300x300
        img = cv2.resize(img, (300, 300))
        
        # Preprocess: Increase brightness and contrast
        alpha = 1.5  # Contrast control (1.0-3.0)
        beta = 50    # Brightness control (0-100)
        img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
        
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
                success_count += 1
        else:
            # Log and save the problematic image
            print(f"No hand detected in {img_path}")
            shutil.copy(img_path, os.path.join(failed_dir, f"{class_name}_{img_name}"))

    print(f"Class {class_name}: {success_count}/500 images successfully processed")

hands.close()
print("Landmark extraction complete. Saved to 'asl_landmarks.csv'")
print(f"Problematic images saved to '{failed_dir}'")
