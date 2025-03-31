import cv2
import mediapipe as mp
import pygame
import math
import sys

# Initialize pygame mixer for sound
pygame.mixer.init()

def play_sound():
    """Play a sound when a push-up is detected."""
    pygame.mixer.init()  # Initialize the mixer
    pygame.mixer.music.load("drum.wav")  # Ensure this file is in the same directory or provide full path
    pygame.mixer.music.play()

# Initialize MediaPipe Pose module
mp_pose = mp.solutions.pose
drawing_utils = mp.solutions.drawing_utils

pushup_count = 0
direction = None  # "down" or "up"

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

def detect_pushup(pose_landmarks):
    """Detect push-ups based on elbow, wrist, and shoulder landmarks."""
    global pushup_count, direction

    shoulder = pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
    elbow = pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW]
    wrist = pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST]

    elbow_angle = calculate_angle(shoulder, elbow, wrist)

    # Going down when elbow angle is decreasing
    if elbow_angle < 100 and direction != "down":
        direction = "down"
    
    # Going up when elbow angle goes back above ~160°
    elif elbow_angle > 160 and direction == "down":
        direction = "up"
        pushup_count += 1
        play_sound()

def draw_pose_results(image, results):
    """Draw elbow, wrist, and shoulder landmarks with lines between them and display the push-up count."""
    global pushup_count
    
    if results.pose_landmarks:
        detect_pushup(results.pose_landmarks)
        
        # Get landmark coordinates
        elbow = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW]
        wrist = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST]
        shoulder = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
        
        h, w, _ = image.shape
        elbow_x, elbow_y = int(elbow.x * w), int(elbow.y * h)
        wrist_x, wrist_y = int(wrist.x * w), int(wrist.y * h)
        shoulder_x, shoulder_y = int(shoulder.x * w), int(shoulder.y * h)
        
        # Draw points and lines
        cv2.circle(image, (elbow_x, elbow_y), 5, (0, 0, 255), -1)  # Red (Elbow)
        cv2.circle(image, (wrist_x, wrist_y), 5, (0, 255, 0), -1)  # Green (Wrist)
        cv2.circle(image, (shoulder_x, shoulder_y), 5, (255, 0, 0), -1)  # Blue (Shoulder)
        cv2.line(image, (elbow_x, elbow_y), (shoulder_x, shoulder_y), (255, 255, 0), 2)
        cv2.line(image, (wrist_x, wrist_y), (elbow_x, elbow_y), (255, 0, 255), 2)
    
    # Display push-up count and controls
    cv2.putText(image, "Press 'a' to reset count", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    cv2.putText(image, "Press 'q' to quit", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    cv2.putText(image, f"Push-up Count: {pushup_count}", (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    
    return image

def run_pushup_counter(video_path='webcam'):
    """Run the push-up counter using a webcam or a video file."""
    global pushup_count
    
    cap = cv2.VideoCapture(0) if video_path == 'webcam' else cv2.VideoCapture(video_path)
    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            break
        
        if video_path == 'webcam':
            image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        results = pose.process(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        image = draw_pose_results(image, results)
        
        # Show the video feed
        cv2.imshow('Push-up Counter', image)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('a'):
            pushup_count = 0
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    """Entry point: Accept a video path argument from the command line."""
    video_path = sys.argv[1] if len(sys.argv) > 1 else 'webcam'
    run_pushup_counter(video_path)
