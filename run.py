import cv2
import pygame
import math
import sys
import numpy as np
import pandas as pd
from ACV_Classifier_Fonctions import Predict,Post_processing

# Initialize pygame mixer for sound
pygame.mixer.init()

pushup_count = 0 # Init pushup_count
previous_position_label = None # Init position_label
sparkle_frames = 0  # Counter for sparkle duration

def play_sound():
    """Play a sound when a push-up is detected."""
    pygame.mixer.init()  # Initialize the mixer
    pygame.mixer.music.load("./media/drum.wav")  # Ensure this file is in the same directory or provide full path
    pygame.mixer.music.play()


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


def image_post_processing_text(image, mode,position_label=None):
    global pushup_count, sparkle_frames, previous_position_label

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

    ppi=Post_processing()
    pushup_model=Predict("pushup_all_DecisionTreeClassifier.pkl")
    yoga_model=Predict("yoga_all_DecisionTreeClassifier.pkl")
    modes = [None, 'push-up', 'yoga']
    mode_index = 0
    run_predict_every_n_frames=15
    frames=0
    
    cap = cv2.VideoCapture(0) if video_path == 'webcam' else cv2.VideoCapture(video_path)
    
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            break

        if video_path == 'webcam':
            image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        
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
        image=ppi.image_post_processing_poselandmark(image)  
        image = image_post_processing_text(image,mode,prediction)

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
