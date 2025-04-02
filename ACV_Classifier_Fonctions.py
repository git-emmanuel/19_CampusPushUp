# ACV Classifier Classes & Fonctions 

import cv2
import mediapipe as mp
import csv
import numpy as np
import os
import math


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


