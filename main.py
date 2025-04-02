import cv2
import mediapipe as mp
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
root.title("Sign Language Recognition")

# Video feed label
video_label = tk.Label(root)
video_label.pack()

# White text area below the video
text_area = tk.Text(root, height=4, width=50, bg="white", fg="black")
text_area.pack(pady=10)  # Add some padding for better layout

def recognize_gesture(hand_landmarks, hand_index=0, all_hands=None):
    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    ring_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
    pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]
    wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]

    # Single-hand signs
    # "B": All fingers up, thumb across palm
    if (index_tip.y < wrist.y and middle_tip.y < wrist.y and 
        ring_tip.y < wrist.y and pinky_tip.y < wrist.y and 
        thumb_tip.x > index_tip.x and thumb_tip.y > wrist.y):
        return "B"
    # "Yes": Index and middle up, others down
    elif (index_tip.y < wrist.y and middle_tip.y < wrist.y and 
          ring_tip.y > wrist.y and pinky_tip.y > wrist.y and 
          thumb_tip.y > wrist.y):
        return "Yes"
    # "No": Index up, others down
    elif (index_tip.y < wrist.y and middle_tip.y > wrist.y and 
          ring_tip.y > wrist.y and pinky_tip.y > wrist.y and 
          thumb_tip.y > wrist.y):
        return "No"
    # "Hello": Index up, thumb down
    elif (index_tip.y < hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].y and 
          thumb_tip.y > hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP].y):
        return "Hello"

    # Two-hand sign: "Mother" (both hands in "M" shape, close together)
    if all_hands and len(all_hands) == 2 and hand_index == 0:
        other_hand = all_hands[1]
        other_index_tip = other_hand.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
        # Check if both hands are in "M" shape (index and middle up, others down, thumb across)
        if (index_tip.y < wrist.y and middle_tip.y < wrist.y and 
            ring_tip.y > wrist.y and pinky_tip.y > wrist.y and 
            thumb_tip.x > middle_tip.x and
            other_index_tip.y < other_hand.landmark[mp_hands.HandLandmark.WRIST].y and 
            other_hand.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y < other_hand.landmark[mp_hands.HandLandmark.WRIST].y and
            other_hand.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].y > other_hand.landmark[mp_hands.HandLandmark.WRIST].y and
            other_hand.landmark[mp_hands.HandLandmark.PINKY_TIP].y > other_hand.landmark[mp_hands.HandLandmark.WRIST].y and
            other_hand.landmark[mp_hands.HandLandmark.THUMB_TIP].x > other_hand.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].x):
            # Check if hands are close together (x-distance between wrists)
            wrist_distance = abs(wrist.x - other_hand.landmark[mp_hands.HandLandmark.WRIST].x)
            if wrist_distance < 0.2:  # Adjust this threshold as needed
                return "Mother"
    
    return "Unknown"

def update_frame():
    ret, frame = cap.read()
    if ret:
        frame = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            interpreted_text = ""
            for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
                mp_draw.draw_landmarks(frame_rgb, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                gesture = recognize_gesture(hand_landmarks, i, results.multi_hand_landmarks)
                interpreted_text += f"Hand {i+1}: {gesture} | "
            text_area.delete(1.0, tk.END)
            text_area.insert(tk.END, interpreted_text.strip())
        else:
            text_area.delete(1.0, tk.END)
            text_area.insert(tk.END, "No hands detected")

        img = Image.fromarray(frame_rgb)
        imgtk = ImageTk.PhotoImage(image=img)
        video_label.imgtk = imgtk
        video_label.configure(image=imgtk)

    root.after(10, update_frame)

update_frame()
root.mainloop()

cap.release()
cv2.destroyAllWindows()