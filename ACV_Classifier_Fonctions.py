# ACV Classifier Classes & Fonctions 

import cv2
import mediapipe as mp
import csv
import numpy as np
import os
import math
import datetime
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import joblib
import kagglehub
import pygame
import time


class MediaPipe:
    def __init__(self):

        # Initialize MediaPipe Pose module
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        
    def import_image(self,image):
        # Define attributs
        self.image = image
        self.results = self.pose.process(image)

    def get_positions(self,image):
        try :
            # Run import image
            self.import_image(image)

            # Define local variables
            landmarks = self.results.pose_landmarks.landmark
            poselandmark=self.mp_pose.PoseLandmark

            # Get key landmarks
            self.positions={}

            # Positions pour le côté gauche
            self.positions['nose'] = landmarks[poselandmark.NOSE]
            self.positions['left_eye_inner'] = landmarks[poselandmark.LEFT_EYE_INNER]
            self.positions['left_eye'] = landmarks[poselandmark.LEFT_EYE]
            self.positions['left_eye_outer'] = landmarks[poselandmark.LEFT_EYE_OUTER]
            self.positions['right_eye_inner'] = landmarks[poselandmark.RIGHT_EYE_INNER]
            self.positions['right_eye'] = landmarks[poselandmark.RIGHT_EYE]
            self.positions['right_eye_outer'] = landmarks[poselandmark.RIGHT_EYE_OUTER]
            self.positions['left_ear'] = landmarks[poselandmark.LEFT_EAR]
            self.positions['right_ear'] = landmarks[poselandmark.RIGHT_EAR]
            self.positions['mouth_left'] = landmarks[poselandmark.MOUTH_LEFT]
            self.positions['mouth_right'] = landmarks[poselandmark.MOUTH_RIGHT]
            self.positions['left_shoulder'] = landmarks[poselandmark.LEFT_SHOULDER]
            self.positions['right_shoulder'] = landmarks[poselandmark.RIGHT_SHOULDER]
            self.positions['left_elbow'] = landmarks[poselandmark.LEFT_ELBOW]
            self.positions['right_elbow'] = landmarks[poselandmark.RIGHT_ELBOW]
            self.positions['left_wrist'] = landmarks[poselandmark.LEFT_WRIST]
            self.positions['right_wrist'] = landmarks[poselandmark.RIGHT_WRIST]
            self.positions['left_pinky'] = landmarks[poselandmark.LEFT_PINKY]
            self.positions['right_pinky'] = landmarks[poselandmark.RIGHT_PINKY]
            self.positions['left_index'] = landmarks[poselandmark.LEFT_INDEX]
            self.positions['right_index'] = landmarks[poselandmark.RIGHT_INDEX]
            self.positions['left_thumb'] = landmarks[poselandmark.LEFT_THUMB]
            self.positions['right_thumb'] = landmarks[poselandmark.RIGHT_THUMB]
            self.positions['left_hip'] = landmarks[poselandmark.LEFT_HIP]
            self.positions['right_hip'] = landmarks[poselandmark.RIGHT_HIP]
            self.positions['left_knee'] = landmarks[poselandmark.LEFT_KNEE]
            self.positions['right_knee'] = landmarks[poselandmark.RIGHT_KNEE]
            self.positions['left_ankle'] = landmarks[poselandmark.LEFT_ANKLE]
            self.positions['right_ankle'] = landmarks[poselandmark.RIGHT_ANKLE]
            self.positions['left_heel'] = landmarks[poselandmark.LEFT_HEEL]
            self.positions['right_heel'] = landmarks[poselandmark.RIGHT_HEEL]
            self.positions['left_foot_index'] = landmarks[poselandmark.LEFT_FOOT_INDEX]
            self.positions['right_foot_index'] = landmarks[poselandmark.RIGHT_FOOT_INDEX]
            
            return self.positions
        
        except:
            None
    
    def get_distances_and_angles(self,image):
        try :
            # Run get positions image
            self.get_positions(image)

            # Get key landmarks
            self.distances_and_angles={}

            # Distances entre les épaules et les hanches
            self.distances_and_angles['left_shoulder_to_left_hip_dist'] = self.calculate_distance(self.positions['left_shoulder'], self.positions['left_hip'])
            self.distances_and_angles['right_shoulder_to_right_hip_dist'] = self.calculate_distance(self.positions['right_shoulder'], self.positions['right_hip'])

            # Distances entre les coudes et les épaules
            self.distances_and_angles['left_elbow_to_left_shoulder_dist'] = self.calculate_distance(self.positions['left_elbow'], self.positions['left_shoulder'])
            self.distances_and_angles['right_elbow_to_right_shoulder_dist'] = self.calculate_distance(self.positions['right_elbow'], self.positions['right_shoulder'])

            # Distances entre les poignets et les coudes
            self.distances_and_angles['left_wrist_to_left_elbow_dist'] = self.calculate_distance(self.positions['left_wrist'], self.positions['left_elbow'])
            self.distances_and_angles['right_wrist_to_right_elbow_dist'] = self.calculate_distance(self.positions['right_wrist'], self.positions['right_elbow'])

            # Distances entre les hanches et les genoux
            self.distances_and_angles['left_hip_to_left_knee_dist'] = self.calculate_distance(self.positions['left_hip'], self.positions['left_knee'])
            self.distances_and_angles['right_hip_to_right_knee_dist'] = self.calculate_distance(self.positions['right_hip'], self.positions['right_knee'])

            # Distances entre les genoux et les chevilles
            self.distances_and_angles['left_knee_to_left_ankle_dist'] = self.calculate_distance(self.positions['left_knee'], self.positions['left_ankle'])
            self.distances_and_angles['right_knee_to_right_ankle_dist'] = self.calculate_distance(self.positions['right_knee'], self.positions['right_ankle'])

            # Distances entre les épaules
            self.distances_and_angles['left_shoulder_to_right_shoulder_dist'] = self.calculate_distance(self.positions['left_shoulder'], self.positions['right_shoulder'])

            # Distances entre les hanches
            self.distances_and_angles['left_hip_to_right_hip_dist'] = self.calculate_distance(self.positions['left_hip'], self.positions['right_hip'])

            # Distances entre les yeux
            self.distances_and_angles['left_eye_to_right_eye_dist'] = self.calculate_distance(self.positions['left_eye'], self.positions['right_eye'])

            # Distances entre le poignet et l'épaule
            self.distances_and_angles['left_wrist_to_left_shoulder_dist'] = self.calculate_distance(self.positions['left_wrist'], self.positions['left_shoulder'])
            self.distances_and_angles['right_wrist_to_right_shoulder_dist'] = self.calculate_distance(self.positions['right_wrist'], self.positions['right_shoulder'])

            # Distances entre le poignet et la hanche
            self.distances_and_angles['left_wrist_to_left_hip_dist'] = self.calculate_distance(self.positions['left_wrist'], self.positions['left_hip'])
            self.distances_and_angles['right_wrist_to_right_hip_dist'] = self.calculate_distance(self.positions['right_wrist'], self.positions['right_hip'])

            # Distances entre le pied et la hanche
            self.distances_and_angles['left_foot_to_left_hip_dist'] = self.calculate_distance(self.positions['left_foot_index'], self.positions['left_hip'])
            self.distances_and_angles['right_foot_to_right_hip_dist'] = self.calculate_distance(self.positions['right_foot_index'], self.positions['right_hip'])

            # Distances entre le pied et l'épaule
            self.distances_and_angles['left_foot_to_left_shoulder_dist'] = self.calculate_distance(self.positions['left_foot_index'], self.positions['left_shoulder'])
            self.distances_and_angles['right_foot_to_right_shoulder_dist'] = self.calculate_distance(self.positions['right_foot_index'], self.positions['right_shoulder'])

            # Distances entre le pied et le poignet
            self.distances_and_angles['left_foot_to_left_wrist_dist'] = self.calculate_distance(self.positions['left_foot_index'], self.positions['left_wrist'])
            self.distances_and_angles['right_foot_to_right_wrist_dist'] = self.calculate_distance(self.positions['right_foot_index'], self.positions['right_wrist'])

            # Distances entre les poignets
            self.distances_and_angles['left_wrist_to_right_wrist_dist'] = self.calculate_distance(self.positions['left_wrist'], self.positions['right_wrist'])

            # Distances entre les coudes
            self.distances_and_angles['left_elbow_to_right_elbow_dist'] = self.calculate_distance(self.positions['left_elbow'], self.positions['right_elbow'])

            # Distances entre les genoux
            self.distances_and_angles['left_knee_to_right_knee_dist'] = self.calculate_distance(self.positions['left_knee'], self.positions['right_knee'])

            # Distances entre les pieds
            self.distances_and_angles['left_foot_to_right_foot_dist'] = self.calculate_distance(self.positions['left_foot_index'], self.positions['right_foot_index'])

            # Distances entre le nez et le poignet
            self.distances_and_angles['nose_left_to_wrist_dist'] = self.calculate_distance(self.positions['nose'], self.positions['left_wrist'])
            self.distances_and_angles['nose_right_to_wrist_dist'] = self.calculate_distance(self.positions['nose'], self.positions['right_wrist'])

            # Distances entre le nez et le pied
            self.distances_and_angles['nose_left_to_foot_dist'] = self.calculate_distance(self.positions['nose'], self.positions['left_foot_index'])
            self.distances_and_angles['nose_right_to_foot_dist'] = self.calculate_distance(self.positions['nose'], self.positions['right_foot_index'])

            # Distances entre le nez et l'épaule
            self.distances_and_angles['nose_left_to_shoulder_dist'] = self.calculate_distance(self.positions['nose'], self.positions['left_shoulder'])
            self.distances_and_angles['nose_right_to_shoulder_dist'] = self.calculate_distance(self.positions['nose'], self.positions['right_shoulder'])

            # Angles pied-genou-hanche
            self.distances_and_angles['left_foot_knee_hip_angle'] = self.calculate_angle(self.positions['left_foot_index'], self.positions['left_knee'], self.positions['left_hip'])
            self.distances_and_angles['right_foot_knee_hip_angle'] = self.calculate_angle(self.positions['right_foot_index'], self.positions['right_knee'], self.positions['right_hip'])

            # Angles épaule-hanche-genou
            self.distances_and_angles['left_shoulder_hip_knee_angle'] = self.calculate_angle(self.positions['left_shoulder'], self.positions['left_hip'], self.positions['left_knee'])
            self.distances_and_angles['right_shoulder_hip_knee_angle'] = self.calculate_angle(self.positions['right_shoulder'], self.positions['right_hip'], self.positions['right_knee'])

            # Angles coude-épaule-hanche
            self.distances_and_angles['left_elbow_shoulder_hip_angle'] = self.calculate_angle(self.positions['left_elbow'], self.positions['left_shoulder'], self.positions['left_hip'])
            self.distances_and_angles['right_elbow_shoulder_hip_angle'] = self.calculate_angle(self.positions['right_elbow'], self.positions['right_shoulder'], self.positions['right_hip'])

            # Angles épaule-coude-poignet
            self.distances_and_angles['left_shoulder_elbow_wrist_angle'] = self.calculate_angle(self.positions['left_shoulder'], self.positions['left_elbow'], self.positions['left_wrist'])
            self.distances_and_angles['right_shoulder_elbow_wrist_angle'] = self.calculate_angle(self.positions['right_shoulder'], self.positions['right_elbow'], self.positions['right_wrist'])

            return self.distances_and_angles
        
        except:
            None
    
    def calculate_distance(self,a,b):
        """Calculate the distance between two points"""
        return np.linalg.norm([a.x - b.x, a.y - b.y])
                
    def calculate_angle(self,a, b, c):
        """Calculate the angle between three points"""
        ba = np.array([a.x - b.x, a.y - b.y])
        bc = np.array([c.x - b.x, c.y - b.y])
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        return np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))


class Embeddings:
    def __init__(self,size=20,embeddings_folder="./embeddings"):
        self.size=size
        self.embeddings_folder=embeddings_folder
        self.mediapipe=MediaPipe()

    def generate_embeddings(self,base_folder,embeddings_filename,features=None,show_images=False,date_generation=False):

        # Créer le dossier s'il n'existe pas
        if not os.path.exists(self.embeddings_folder):
            os.makedirs(self.embeddings_folder)

        if date_generation:
            # Obtenir la date et l'heure actuelles
            now = datetime.datetime.now()
            timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")

            # Nom du fichier avec la date et l'heure
            embeddings_filename = f"{embeddings_filename}_embeddings_{timestamp}.csv"
        else:
            embeddings_filename = f"{embeddings_filename}_embeddings.csv"
        
        embeddings_pathfile=os.path.join(self.embeddings_folder,embeddings_filename)

        # Set defaut features
        if features==None:
            ref_image = cv2.imread('./training_datasets/ref_image.jpg') # Load image
            ref_image = cv2.cvtColor(ref_image, cv2.COLOR_BGR2RGB) # Convert to RGB
            ref_image=self.mediapipe.get_distances_and_angles(ref_image) # Get distance and angles
            features=list(ref_image.keys()) # Gets keys and add to list
            features.append('label') #
            features_list_pathfile=os.path.join(self.embeddings_folder,'features_list.csv')
            with open(features_list_pathfile, "w", newline="") as file:
                writer = csv.writer(file)
                for key in features:
                    writer.writerow([key])

        if not 'label' in features:
            features.append('label')

        # Create CSV file and write headers"
        with open(embeddings_pathfile, "w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(features)
            
            count_processed=0
            count_skipped=0

            for label in os.listdir(base_folder):
                label_folder = os.path.join(base_folder, label)
                if not os.path.isdir(label_folder):
                    continue

                for image_name in os.listdir(label_folder):
                    image_path = os.path.join(label_folder, image_name)
                        
                    image = cv2.imread(image_path)

                    try:
                        # Convert images to RGB
                        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                        # Show image processed
                        if show_images:
                            cv2.imshow("Pose Capture", image)
                            if cv2.waitKey(50) & 0xFF == ord('q'):
                                break

                        # Get distances and angles from mediapipe
                        image_data=self.mediapipe.get_distances_and_angles(image_rgb)

                        # Generate embedding data
                        csv_data=[]
                        for feature in features:
                            try:
                                csv_data.append(image_data[feature])
                            except:
                                None
                        csv_data.append(label)
                            
                        # Save embeddings to CSV
                        writer.writerow(csv_data)
                        # print(f"Image processed : {image_name}")

                        count_processed +=1
                        
                    except:
                        count_skipped +=1
                        # print(f"Image not processed (no images or error) : {image_name}")
                
        cv2.destroyAllWindows()
        
        print(f"Images processed : {count_processed}")
        print(f"Images skipped : {count_skipped}")
        print(f'CSV pathfile generated :{embeddings_pathfile}')

        # Add full results in embeddings dataset
        embeddings=pd.read_csv(embeddings_pathfile)
        return embeddings
    
    def import_from_kaggle(self,kaggle_path):

        # Download latest version
        base_folder = kagglehub.dataset_download(kaggle_path)
        print("Path to dataset files:", base_folder)
        
        return base_folder

class Classifier:
    def __init__(self,embedding_folders='./embeddings',models_folder='./models'):
        self.embedding_folders=embedding_folders
        self.models_folder=models_folder
        
    def fit_and_save_model(self,model,embeddings_filename):

        # Créer le dossier s'il n'existe pas
        if not os.path.exists(self.models_folder):
            os.makedirs(self.models_folder)

        embeddings=pd.read_csv(os.path.join(self.embedding_folders,embeddings_filename))

        # Split X,y
        X = embeddings.drop(columns=['label'])  # Features: distances
        y = embeddings['label']  # Target: position labels

        # Train classifier
        model.fit(X, y)

        # Define model classifier name
        model_filename = embeddings_filename.replace('embeddings.csv',f'{model.__class__.__name__}.pkl')

        # Save model
        joblib.dump(model,os.path.join(self.models_folder,model_filename))

        return model

class Predict:
    def __init__(self,model_filename,models_folder='./models'):

        # Load a trained model
        self.model= joblib.load(os.path.join(models_folder,model_filename)) 

        # Start mediapipe instance
        self.mp=MediaPipe()

    def load_image(self,image_path):
        # Load image
        image = cv2.imread(image_path)

        # Convert images to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        return image_rgb
    
    def predict(self,image):
        try :
            # Get distances and angles from mediapipe
            image_data=self.mp.get_distances_and_angles(image)

            # Convert to df
            image_data=pd.DataFrame(image_data,index=['value'])

            # Filter by model features
            image_data=image_data.loc[:,self.model.feature_names_in_]
        
            # Predict based on image_data
            return self.model.predict(image_data)[0]
        except : 
            return None

    
    def mean_predict(self,prediction,min_predictions=20):

        predictions=[]
        if len(predictions)>=min_predictions:
            predictions.pop[0]
        if len(predictions)<min_predictions:
            if prediction is not None:
                predictions.append(prediction)

        # Trouver l'élément le plus présent
        most_frequent_prediction = max(set(predictions), key=predictions.count)

        return most_frequent_prediction
    

class Post_processing:
    def __init__(self):
        self.ppi_mediapipe=MediaPipe()
        self.pushup_logo = cv2.imread("media/push-up_logo.png", cv2.IMREAD_UNCHANGED)
        self.pushup_logo=cv2.resize(self.pushup_logo, (800,800))
        self.yoga_logo = cv2.imread("media/yoga_logo.png", cv2.IMREAD_UNCHANGED)
        self.yoga_logo=cv2.resize(self.yoga_logo, (1400,800))
        self.pushup_count = 0 # Init pushup_count
        self.previous_position_label = None # Init position_label
        self.sparkle_frames = 0  # Counter for sparkle duration
        pygame.mixer.init()  # Initialize the mixer
        pygame.mixer.music.load("./media/drum.wav")  # Ensure this file is in the same directory or provide full path
        self.prev_time = time.time()
        self.fps_values=[]
        self.load_yoga_images()

        
    def image_post_processing_poselandmark(self,image):
        """Draw elbow, wrist, and shoulder landmarks with lines between them and display the push-up count."""
        
        # Get Image shape
        h, w, _ = image.shape
        circle_color=(0, 0, 255)
        circle_size=3
        line_color=(0, 255, 0)
        line_thickness=2
        
        try : 
            # Get positions
            positions=self.ppi_mediapipe.get_positions(image)

            # for each position, adapt to image shape and trace circle
            positions_shaped={}
            for position in positions:
                positions_shaped[position]={'x':int(positions[position].x * w),'y': int(positions[position].y * h)}
                cv2.circle(image, (positions_shaped[position]['x'], positions_shaped[position]['y']), circle_size, circle_color, -1)
            
           
            # Dessiner les lignes
            cv2.line(image, (positions_shaped['left_elbow']['x'], positions_shaped['left_elbow']['y']),
                    (positions_shaped['left_shoulder']['x'], positions_shaped['left_shoulder']['y']), line_color, line_thickness)

            cv2.line(image, (positions_shaped['left_wrist']['x'], positions_shaped['left_wrist']['y']),
                    (positions_shaped['left_elbow']['x'], positions_shaped['left_elbow']['y']), line_color, line_thickness)

            cv2.line(image, (positions_shaped['right_elbow']['x'], positions_shaped['right_elbow']['y']),
                    (positions_shaped['right_shoulder']['x'], positions_shaped['right_shoulder']['y']), line_color, line_thickness)

            cv2.line(image, (positions_shaped['right_wrist']['x'], positions_shaped['right_wrist']['y']),
                    (positions_shaped['right_elbow']['x'], positions_shaped['right_elbow']['y']), line_color, line_thickness)

            cv2.line(image, (positions_shaped['left_shoulder']['x'], positions_shaped['left_shoulder']['y']),
                    (positions_shaped['right_shoulder']['x'], positions_shaped['right_shoulder']['y']), line_color, line_thickness)

            cv2.line(image, (positions_shaped['left_hip']['x'], positions_shaped['left_hip']['y']),
                    (positions_shaped['left_shoulder']['x'], positions_shaped['left_shoulder']['y']), line_color, line_thickness)

            cv2.line(image, (positions_shaped['right_hip']['x'], positions_shaped['right_hip']['y']),
                    (positions_shaped['right_shoulder']['x'], positions_shaped['right_shoulder']['y']), line_color, line_thickness)

            cv2.line(image, (positions_shaped['left_hip']['x'], positions_shaped['left_hip']['y']),
                    (positions_shaped['left_knee']['x'], positions_shaped['left_knee']['y']), line_color, line_thickness)

            cv2.line(image, (positions_shaped['right_hip']['x'], positions_shaped['right_hip']['y']),
                    (positions_shaped['right_knee']['x'], positions_shaped['right_knee']['y']), line_color, line_thickness)

            cv2.line(image, (positions_shaped['left_knee']['x'], positions_shaped['left_knee']['y']),
                    (positions_shaped['left_foot_index']['x'], positions_shaped['left_foot_index']['y']), line_color, line_thickness)

            cv2.line(image, (positions_shaped['right_knee']['x'], positions_shaped['right_knee']['y']),
                    (positions_shaped['right_foot_index']['x'], positions_shaped['right_foot_index']['y']), line_color, line_thickness)
            
        except:
            None
        return image
    
    def image_post_processing_logo(self,image, mode,prediction=None):
        try : 
            # Add logo to the image at the top right corner
            match mode:
                case 'push-up':
                    logo = self.pushup_logo
                case 'yoga':
                    try:
                        logo=self.yoga_images[prediction]
                    except :
                        logo = self.yoga_logo
                case _:
                    logo = None

            if logo is not None:
                image = overlay_image(image, logo, 5, 5,scale=0.15)
        except : 
            None
        return image
    

    def image_post_processing_text(self,image, mode,position_label=None):
        try : 
            if mode=='yoga':
                position_label=self.yoga_positions['Francais'][self.yoga_positions['Anglais']==position_label].values[0]

            # Display push-up count and controls with black text on a white stripe
            draw_text_with_background(image, "Press 'r' to reset count", (5, 20))
            draw_text_with_background(image, "Press 'q' to quit", (5, 40))
            draw_text_with_background(image, "Press 'f' for full screen", (5, 60))
            draw_text_with_background(image, f"Mode (press 'm' to toggle): {mode}", (5, 80))
            draw_text_with_background(image, f"AI classification: {position_label}", (5, 100))

            # Add counter
            if mode=='push-up':
                # Update counter
                if position_label=='position_up' and self.previous_position_label=='position_down':
                    self.pushup_count +=1
                    self.sparkle_frames = 10  # Trigger sparkles for next 10 frames
                    pygame.mixer.music.play()
                self.previous_position_label=position_label
                draw_text_with_background(image, f"Push-up Count: {self.pushup_count}", (5, 140))
        except : 
            None
        return image
    
    def image_post_processing_sparkles(self,image):
        try : 
            # Show sparkles only for 10 frames after a push-up is detected
            if self.sparkle_frames > 0:
                image = draw_sparkles(image)
                self.sparkle_frames -= 1  # Decrease count each frame
        except : 
            None
        return image
    
    def image_post_processing_fps(self,image):
        try : 
            # Displays FPS
            curr_time = time.time()
            if len(self.fps_values)>10:
                self.fps_values.pop(0)
            if len(self.fps_values)<10:
                self.fps_values.append(1 / (curr_time - self.prev_time))
            draw_text_with_background(image, f"FPS: {int(np.mean(self.fps_values))}", (5, 120))
            self.prev_time = curr_time
        except : 
            None
        return image
    
    def load_yoga_images(self):
        self.yoga_positions=pd.read_csv('./embeddings/yoga_positions.csv')
        
        self.yoga_images={}
        yoga_folder='./media/yoga_images'
        for image_name in os.listdir(yoga_folder):
            yoga_image = cv2.imread(os.path.join(yoga_folder, image_name), cv2.IMREAD_UNCHANGED)
            self.yoga_images['{}'.format(os.path.splitext(image_name)[0])] = cv2.resize(yoga_image, (1400,800))
            
def overlay_image(background, overlay, x, y, scale=0.1):
    """Overlay an image onto another at a specified position with scaling."""
    h, w, _ = background.shape
    overlay = cv2.resize(overlay, (int(overlay.shape[1] * scale), int(overlay.shape[0] * scale)))
    oh, ow, _ = overlay.shape

    # Calculer la position x, y pour placer l'overlay à une distance spécifique du coin supérieur droit
    x = background.shape[1] - ow - x

    if x < 0:
        x = 0
    if y < 0:
        y = 0
    if x + ow > background.shape[1]:
        x = background.shape[1] - ow
    if y + oh > background.shape[0]:
        y = background.shape[0] - oh
    
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
    font_scale = 0.5
    font_thickness = 1
    text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
    x, y = position
    cv2.rectangle(image, (x - 5, y - text_size[1] - 5), (x + text_size[0] + 5, y + 5), (255, 255, 255), -1)
    cv2.putText(image, text, (x, y), font, font_scale, (0, 0, 0), font_thickness)

