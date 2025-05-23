import cv2
import sys
from ACV_Classifier_Fonctions import Predict,Post_processing

def run_video_capture(video_path='webcam'):
    """Run the video capture using a webcam or a video file."""

    # Init classes and variables
    ppi=Post_processing()
    pushup_model=Predict("pushup_all_DecisionTreeClassifier.pkl")
    yoga_model=Predict("yoga_all_DecisionTreeClassifier.pkl")
    modes = [None, 'push-up', 'yoga']
    mode_index = 0
    run_predict_every_n_frames=6
    frames=0
    fullscreen=False
    model=None

    # Create a window
    cv2.namedWindow('Video_Capture', cv2.WND_PROP_FULLSCREEN)
    cv2.resizeWindow('Video_Capture', 800, 600)  # Définir la taille initiale de la fenêtre
    
    # Define video capture
    cap = cv2.VideoCapture(0) if video_path == 'webcam' else cv2.VideoCapture(video_path)
    
    # Run video capture
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
        match mode:
            case 'push-up':
                model = pushup_model
            case 'yoga':
                model = yoga_model
            case _:
                model = None
                prediction = None
        # Model Prediction
        frames+=1
        if frames >= run_predict_every_n_frames:
            frames=0
            if model is not None : 
                prediction = model.predict(image)
                prediction = model.mean_predict(prediction)


        # Image post-processing 
        image = ppi.image_post_processing_poselandmark(image)  
        image = ppi.image_post_processing_logo(image,mode,prediction)
        image = ppi.image_post_processing_text(image,mode,prediction)
        image = ppi.image_post_processing_sparkles(image)
        image = ppi.image_post_processing_fps(image)

        # Set RGB
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Resize image size
        if fullscreen : 
            image = cv2.resize(image, (1400,1000))
        else:
            image = cv2.resize(image, (800, 600))

        # Show the video feed
        cv2.imshow('Video_Capture', image)
        
        key = cv2.waitKey(1) & 0xFF

        # Quit
        if key == ord('q'):
            break

        # Reset counter
        elif key == ord('r'):
            ppi.pushup_count=0

        # Toggle detection mode     
        elif key == ord('m'):
            mode_index = (mode_index + 1) % len(modes)

         # Toggle fullscreen
        elif key == ord('f'):
            fullscreen = not fullscreen
            cv2.setWindowProperty('Video_Capture', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN if fullscreen else cv2.WINDOW_NORMAL)

    
    # Close video capture
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    """Entry point: Accept a video path argument from the command line."""
    video_path = sys.argv[1] if len(sys.argv) > 1 else 'webcam'
    run_video_capture(video_path)
