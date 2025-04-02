import cv2
import mediapipe as mp
import csv
import tkinter as tk
from PIL import Image, ImageTk
import numpy as np

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Set up the webcam
cap = cv2.VideoCapture(0)

# Set up the GUI
root = tk.Tk()
root.title("Data Collection for Sign Language")

# Video feed label
video_label = tk.Label(root)
video_label.pack()

# Entry for sign label
label_entry = tk.Entry(root, width=20)
label_entry.pack(pady=5)
label_entry.insert(0, "Enter sign (e.g., A, B, Yes)")

# Button to save data
def save_data():
    sign_label = label_entry.get()
    if sign_label and current_landmarks:
        with open("sign_data.csv", "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([sign_label] + current_landmarks)
        print(f"Saved data for '{sign_label}'")
        text_area.delete(1.0, tk.END)
        text_area.insert(tk.END, f"Saved: {sign_label}")

save_button = tk.Button(root, text="Save Gesture", command=save_data)
save_button.pack(pady=5)

# White text area
text_area = tk.Text(root, height=4, width=50, bg="white", fg="black")
text_area.pack(pady=10)

current_landmarks = []

def update_frame():
    global current_landmarks
    ret, frame = cap.read()
    if ret:
        frame = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            text = ""
            current_landmarks = []
            for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
                mp_draw.draw_landmarks(frame_rgb, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                landmarks = [coord for landmark in hand_landmarks.landmark for coord in (landmark.x, landmark.y, landmark.z)]
                if len(results.multi_hand_landmarks) == 1:
                    landmarks.extend([0] * 63)  # Pad with zeros for second hand
                elif i == 1:
                    current_landmarks.extend(landmarks)  # Second hand
                else:
                    current_landmarks = landmarks  # First hand
                text += f"Hand {i+1}: Detected | "
            text_area.delete(1.0, tk.END)
            text_area.insert(tk.END, text.strip())
        else:
            text_area.delete(1.0, tk.END)
            text_area.insert(tk.END, "No hands detected")
            current_landmarks = []

        img = Image.fromarray(frame_rgb)
        imgtk = ImageTk.PhotoImage(image=img)
        video_label.imgtk = imgtk
        video_label.configure(image=imgtk)

    root.after(10, update_frame)

# Create CSV file with headers if it doesn’t exist
try:
    with open("sign_data.csv", "x", newline="") as f:
        writer = csv.writer(f)
        headers = ["label"] + [f"hand1_{i}_{coord}" for i in range(21) for coord in ("x", "y", "z")] + \
                            [f"hand2_{i}_{coord}" for i in range(21) for coord in ("x", "y", "z")]
        writer.writerow(headers)
except FileExistsError:
    pass

update_frame()
root.mainloop()

cap.release()
cv2.destroyAllWindows()