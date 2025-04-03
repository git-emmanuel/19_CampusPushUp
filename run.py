import cv2
import mediapipe as mp
import pygame
import math
import sys
import time
import numpy as np
import joblib
import pandas as pd
from ACV_Classifier_Fonctions import Predict

# Initialize pygame mixer for sound
pygame.mixer.init()

# Initialize MediaPipe Pose module
mp_pose = mp.solutions.pose


pushup_count = 0 # Init pushup_count
previous_position_label = None # Init position_label
sparkle_frames = 0  # Counter for sparkle duration

def play_sound():
    """Play a sound when a push-up is detected."""
    pygame.mixer.init()  # Initialize the mixer
    pygame.mixer.music.load("./media/drum.wav")  # Ensure this file is in the same directory or provide full path
    pygame.mixer.music.play()


def calculate_angle(a, b, c):
    """Calculate the angle between three points: a (shoulder), b (elbow), and c (wrist)."""
    ba = (a.x - b.x, a.y - b.y)  # Vector BA
    bc = (c.x - b.x, c.y - b.y)  # Vector BC

    dot_product = ba[0] * bc[0] + ba[1] * bc[1]
    magnitude_ba = math.sqrt(ba[0] ** 2 + ba[1] ** 2)
    magnitude_bc = math.sqrt(bc[0] ** 2 + bc[1] ** 2)

    if magnitude_ba * magnitude_bc == 0:
        return 0  # Avoid division by zero

    angle = math.degrees(math.acos(dot_product / (magnitude_ba * magnitude_bc)))
    return angle

def overlay_image(background, overlay, x, y, scale=0.1):
    """Overlay an image onto another at a specified position with scaling."""
    h, w, _ = background.shape
    overlay = cv2.resize(overlay, (int(overlay.shape[1] * scale), int(overlay.shape[0] * scale)))
    oh, ow, _ = overlay.shape
    x, y = w - ow - 20, 20  # Position at the top right
    
    if overlay.shape[2] == 4:  # Handle transparency
        alpha_channel = overlay[:, :, 3] / 255.0
        for c in range(3):
            background[y:y+oh, x:x+ow, c] = (1 - alpha_channel) * background[y:y+oh, x:x+ow, c] + alpha_channel * overlay[:, :, c]
    else:
        background[y:y+oh, x:x+ow] = overlay
    return background


def draw_sparkles(image):
    """Draw sparkles at random positions on the image."""
    h, w, _ = image.shape
    for _ in range(50):  # More sparkles
        x, y = np.random.randint(0, w), np.random.randint(0, h)
        cv2.circle(image, (x, y), 4, (255, 255, 255), -1)  # White small dots
        x, y = np.random.randint(0, w), np.random.randint(0, h)
        color = (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255))  # Slightly yellowish-white
        cv2.circle(image, (x, y), np.random.randint(3, 6), color, -1)
    return image


def draw_text_with_background(image, text, position):
    """Draw black text on a white background strip."""
    font = cv2.FONT_HERSHEY_DUPLEX
    font_scale = 0.7
    font_thickness = 1
    text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
    x, y = position
    cv2.rectangle(image, (x - 5, y - text_size[1] - 5), (x + text_size[0] + 5, y + 5), (255, 255, 255), -1)
    cv2.putText(image, text, (x, y), font, font_scale, (0, 0, 0), font_thickness)


def draw_pose_results(image, mode,results=None,position_label=None):
    """Draw elbow, wrist, and shoulder landmarks with lines between them and display the push-up count."""
    global pushup_count, sparkle_frames, previous_position_label
    
    if results.pose_landmarks:
    
        # Get landmark coordinates for left arm
        elbow_left = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW]
        wrist_left = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST]
        shoulder_left = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]

        # Get landmark coordinates for right arm
        elbow_right = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW]
        wrist_right = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST]
        shoulder_right = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        
        # Get more landmarks
        hip_left = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP]
        hip_right = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP]
        knee_left = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE]
        knee_right = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_KNEE]
        foot_left = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HEEL]
        foot_right = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HEEL]

        h, w, _ = image.shape
        
        # Draw points and lines for left arm
        elbow_left_x, elbow_left_y = int(elbow_left.x * w), int(elbow_left.y * h)
        wrist_left_x, wrist_left_y = int(wrist_left.x * w), int(wrist_left.y * h)
        shoulder_left_x, shoulder_left_y = int(shoulder_left.x * w), int(shoulder_left.y * h)
        cv2.circle(image, (elbow_left_x, elbow_left_y), 5, (0, 0, 255), -1)  # Red (Elbow)
        cv2.circle(image, (wrist_left_x, wrist_left_y), 5, (0, 255, 0), -1)  # Green (Wrist)
        cv2.circle(image, (shoulder_left_x, shoulder_left_y), 5, (255, 0, 0), -1)  # Blue (Shoulder)
        cv2.line(image, (elbow_left_x, elbow_left_y), (shoulder_left_x, shoulder_left_y), (255, 255, 0), 2)
        cv2.line(image, (wrist_left_x, wrist_left_y), (elbow_left_x, elbow_left_y), (255, 0, 255), 2)

        # Draw points and lines for right arm
        elbow_right_x, elbow_right_y = int(elbow_right.x * w), int(elbow_right.y * h)
        wrist_right_x, wrist_right_y = int(wrist_right.x * w), int(wrist_right.y * h)
        shoulder_right_x, shoulder_right_y = int(shoulder_right.x * w), int(shoulder_right.y * h)
        cv2.circle(image, (elbow_right_x, elbow_right_y), 5, (0, 0, 255), -1)  # Red (Elbow)
        cv2.circle(image, (wrist_right_x, wrist_right_y), 5, (0, 255, 0), -1)  # Green (Wrist)
        cv2.circle(image, (shoulder_right_x, shoulder_right_y), 5, (255, 0, 0), -1)  # Blue (Shoulder)
        cv2.line(image, (elbow_right_x, elbow_right_y), (shoulder_right_x, shoulder_right_y), (255, 255, 0), 2)
        cv2.line(image, (wrist_right_x, wrist_right_y), (elbow_right_x, elbow_right_y), (255, 0, 255), 2)

        # Draw lines for shoulders
        cv2.line(image, (shoulder_left_x, shoulder_left_y), (shoulder_right_x, shoulder_right_y), (255, 0, 255), 2)

        # Draw more markers
        hip_right_x, hip_right_y = int(hip_right.x * w), int(hip_right.y * h)
        hip_left_x, hip_left_y = int(hip_left.x * w), int(hip_left.y * h)
        knee_right_x, knee_right_y = int(knee_right.x * w), int(knee_right.y * h)
        knee_left_x, knee_left_y = int(knee_left.x * w), int(knee_left.y * h)
        cv2.circle(image, (hip_left_x, hip_left_y), 5, (0, 255, 0), -1)  
        cv2.circle(image, (hip_right_x, hip_right_y), 5, (0, 255, 0), -1)
        cv2.circle(image, (knee_left_x, knee_left_y), 5, (0, 255, 0), -1)  
        cv2.circle(image, (knee_right_x, knee_right_y), 5, (0, 255, 0), -1)  
        cv2.line(image, (hip_left_x, hip_left_y), (shoulder_left_x, shoulder_left_y), (255, 0, 255), 2)
        cv2.line(image, (hip_right_x, hip_right_y), (shoulder_right_x, shoulder_right_y), (255, 0, 255), 2)
        cv2.line(image, (hip_left_x, hip_left_y), (knee_left_x, knee_left_y), (255, 0, 255), 2)
        cv2.line(image, (hip_right_x, hip_right_y), (knee_right_x, knee_right_y), (255, 0, 255), 2)

        # Now, add points for the feet
        foot_left_x, foot_left_y = int(foot_left.x * w), int(foot_left.y * h)  # Left foot
        foot_right_x, foot_right_y = int(foot_right.x * w), int(foot_right.y * h)  # Right foot
        # Draw the feet points
        cv2.circle(image, (foot_left_x, foot_left_y), 5, (255, 255, 0), -1)  # Cyan (Left foot)
        cv2.circle(image, (foot_right_x, foot_right_y), 5, (255, 255, 0), -1)  # Cyan (Right foot)

        # Draw lines for feet connection 
        cv2.line(image, (knee_left_x, knee_left_y), (foot_left_x, foot_left_y), (255, 0, 0), 2)  # Left leg line
        cv2.line(image, (knee_right_x, knee_right_y), (foot_right_x, foot_right_y), (255, 0, 0), 2)  # Right leg line

    # Display push-up count and controls with black text on a white stripe
    draw_text_with_background(image, "Press 'r' to reset count", (30, 30))
    draw_text_with_background(image, "Press 'q' to quit", (30, 60))
    draw_text_with_background(image, "Press 'f' for full screen", (30, 90))
    draw_text_with_background(image, f"Mode (press 'm' to toggle): {mode}", (30, 120))
    draw_text_with_background(image, f"AI classification: {position_label}", (30, 150))

    # Add counter
    if mode=='push-up':
        # Update counter
        if position_label=='position_up' and position_label!=previous_position_label and previous_position_label!=None:
            pushup_count +=1
            sparkle_frames = 10  # Trigger sparkles for next 10 frames
            # play_sound()
        previous_position_label=position_label
        draw_text_with_background(image, f"Push-up Count: {pushup_count}", (30, 180))

    # Add logo to the image at the top right corner
    match mode:
        case 'push-up':
            logo = cv2.imread("media/push-up_logo.png", cv2.IMREAD_UNCHANGED)
        case 'yoga':
            logo = cv2.imread("media/yoga_logo.png", cv2.IMREAD_UNCHANGED)
        case _:
            logo = None

    if logo is not None:
        image = overlay_image(image, logo, 10, 10)

    # Show sparkles only for 10 frames after a push-up is detected
    if sparkle_frames > 0:
        image = draw_sparkles(image)
        sparkle_frames -= 1  # Decrease count each frame

    return image


def run_video_capture(video_path='webcam'):
    """Run the video capture using a webcam or a video file."""
    global pushup_count, mode
    
    cap = cv2.VideoCapture(0) if video_path == 'webcam' else cv2.VideoCapture(video_path)
    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    
    pushup_model=Predict("pushup_all_DecisionTreeClassifier.pkl")
    yoga_model=Predict("yoga_all_DecisionTreeClassifier.pkl")
    modes = [None, 'push-up', 'yoga']
    mode_index = 0
    run_predict_every_n_frames=15
    frames=0
    
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            break

        if video_path == 'webcam':
            image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        results = pose.process(image)
        
        # Mode affiliation
        mode=modes[mode_index]

        # Model Prediction
        frames+=1
        if frames >= run_predict_every_n_frames:
            frames=0
            
            match mode:
                case 'push-up':
                    prediction = pushup_model.mean_predict(image,min_predictions=1)
                case 'yoga':
                    prediction = yoga_model.mean_predict(image,min_predictions=1)
                case _:
                    prediction = None
            previous_prediction=prediction
        else : 
            try :
                prediction = previous_prediction
            except :
                prediction = None

        # Image post-processing   
        image = draw_pose_results(image,mode,results,position_label=prediction)

        # Show the video feed
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imshow('Video_Capture', image)
        
        key = cv2.waitKey(1) & 0xFF

        # Quit
        if key == ord('q'):
            break

        # Reset counter
        elif key == ord('r'):
            pushup_count = 0

        # Toggle detection mode     
        elif key == ord('m'):
            mode_index = (mode_index + 1) % len(modes)
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    """Entry point: Accept a video path argument from the command line."""
    video_path = sys.argv[1] if len(sys.argv) > 1 else 'webcam'
    run_video_capture(video_path)
