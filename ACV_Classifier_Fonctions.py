# ACV Classifier Classes & Fonctions 

import cv2
import mediapipe as mp
import csv
import numpy as np
import os
import math
import datetime


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
            self.positions['left_shoulder'] = landmarks[poselandmark.LEFT_SHOULDER]
            self.positions['left_hip'] = landmarks[poselandmark.LEFT_HIP]
            self.positions['left_knee'] = landmarks[poselandmark.LEFT_KNEE]
            self.positions['left_elbow'] = landmarks[poselandmark.LEFT_ELBOW]
            self.positions['left_wrist'] = landmarks[poselandmark.LEFT_WRIST]

            # Positions pour le côté droit
            self.positions['right_shoulder'] = landmarks[poselandmark.RIGHT_SHOULDER]
            self.positions['right_hip'] = landmarks[poselandmark.RIGHT_HIP]
            self.positions['right_knee'] = landmarks[poselandmark.RIGHT_KNEE]
            self.positions['right_elbow'] = landmarks[poselandmark.RIGHT_ELBOW]
            self.positions['right_wrist'] = landmarks[poselandmark.RIGHT_WRIST]
            
            return self.positions
        
        except:
            None
    
    def get_distances_and_angles(self,image):
        try :
            # Run get positions image
            self.get_positions(image)

            # Get key landmarks
            self.distances_and_angles={}

            # Calculs pour le côté gauche
            self.distances_and_angles['left_shoulder_hip_dist'] = self.calculate_distance(self.positions['left_shoulder'], self.positions['left_hip'])
            self.distances_and_angles['left_hip_knee_dist'] = self.calculate_distance(self.positions['left_hip'], self.positions['left_knee'])
            self.distances_and_angles['left_elbow_angle'] = self.calculate_angle(self.positions['left_shoulder'], self.positions['left_elbow'], self.positions['left_wrist'])

            # Calculs pour le côté droit
            self.distances_and_angles['right_shoulder_hip_dist'] = self.calculate_distance(self.positions['right_shoulder'], self.positions['right_hip'])
            self.distances_and_angles['right_hip_knee_dist'] = self.calculate_distance(self.positions['right_hip'], self.positions['right_knee'])
            self.distances_and_angles['right_elbow_angle'] = self.calculate_angle(self.positions['right_shoulder'], self.positions['right_elbow'], self.positions['right_wrist'])
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
    def __init__(self,size=20,output_folder="./embeddings"):
        self.size=size
        self.output_folder=output_folder

        
        self.mediapipe=MediaPipe()

    def generate_embeddings(self,base_folder,features,embeddings_filename,show_images=True,date_generation=False):

        # Créer le dossier s'il n'existe pas
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)

        if date_generation:
            # Obtenir la date et l'heure actuelles
            now = datetime.datetime.now()
            timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")

            # Nom du fichier avec la date et l'heure
            embeddings_filename = f"{embeddings_filename}_embeddings_{timestamp}.csv"
        else:
            embeddings_filename = f"{embeddings_filename}_embeddings.csv"
        
        # Create CSV file and write headers"
        with open(os.path.join(self.output_folder,embeddings_filename), "w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(features)
            
            self.embeddings={}
            count_processed=0
            count_not_processed=0

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
                            if cv2.waitKey(500) & 0xFF == ord('q'):
                                break

                        # Get distances and angles from mediapipe
                        image_data=self.mediapipe.get_distances_and_angles(image_rgb)

                        # Add full results in embeddings dataset
                        self.embeddings[f'{label}_{image_name}']=image_data
                        
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
                        count_not_processed +=1
                        # print(f"Image not processed (no images or error) : {image_name}")
                
        cv2.destroyAllWindows()
        
        print(f"Images processed : {count_processed}")
        print(f"Images not processed : {count_not_processed}")

        return self.embeddings
