"""
Drone Detection System using YOLO11
Author: Alok Kumar
Description: Real-time drone detection system using YOLO11 model with webcam integration
             and audio alert functionality.
"""

import cv2
import numpy as np
import torch
import threading
import argparse
import os
from playsound import playsound
from ultralytics import YOLO


class DroneDetector:
    """
    A class for detecting drones in real-time using YOLO11 model.

    Attributes:
        model_path (str): Path to the trained YOLO11 model weights
        alert_sound (str): Path to the audio alert file
        confidence_threshold (float): Minimum confidence score for detection
        video_source (int or str): Video source (0 for webcam or path to video file)
    """

    def __init__(self, model_path, alert_sound=None, confidence_threshold=0.6, video_source=0):
        """
        Initialize the DroneDetector with specified parameters.

        Args:
            model_path (str): Path to YOLO11 model weights
            alert_sound (str, optional): Path to alert sound file
            confidence_threshold (float): Detection confidence threshold (default: 0.6)
            video_source (int or str): Video input source (default: 0 for webcam)
        """
        self.model_path = model_path
        self.alert_sound = alert_sound
        self.confidence_threshold = confidence_threshold
        self.video_source = video_source
        self.alert_triggered = False
        self.model = None
        self.cap = None

    def load_model(self):
        """
        Load the YOLO11 model from the specified path.

        Returns:
            bool: True if model loaded successfully, False otherwise
        """
        try:
            self.model = YOLO(self.model_path)
            print(f"Model loaded successfully from {self.model_path}")
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False

    def initialize_camera(self):
        """
        Initialize the video capture device.

        Returns:
            bool: True if camera initialized successfully, False otherwise
        """
        self.cap = cv2.VideoCapture(self.video_source)

        if not self.cap.isOpened():
            print("Error: Could not open video source.")
            return False

        print("Camera initialized successfully.")
        return True

    def play_alert(self):
        """
        Play alert sound in a separate thread to avoid blocking detection.
        """
        if self.alert_sound and os.path.exists(self.alert_sound):
            playsound(self.alert_sound)
        self.alert_triggered = False

    def detect_and_annotate(self, frame):
        """
        Perform drone detection on a frame and annotate detected objects.

        Args:
            frame (numpy.ndarray): Input frame from video source

        Returns:
            tuple: (annotated_frame, detection_flag)
                - annotated_frame: Frame with bounding boxes and labels
                - detection_flag: True if drone detected, False otherwise
        """
        # Run YOLO inference
        results = self.model.predict(frame, conf=self.confidence_threshold, verbose=False)

        detected = False

        for result in results:
            # Extract detection information
            boxes = result.boxes.xyxy.cpu().numpy() if result.boxes else []
            confs = result.boxes.conf.cpu().numpy() if result.boxes else []
            cls = result.boxes.cls.cpu().numpy() if result.boxes else []

            # Process each detection
            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = map(int, box)
                confidence = confs[i]
                class_id = int(cls[i])

                if confidence > 0.4:  # Additional confidence check
                    detected = True

                    # Draw bounding box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                    # Add label with confidence score
                    label = f"Drone {confidence:.2f}"
                    cv2.putText(frame, label, (x1, y1 - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        return frame, detected

    def run(self):
        """
        Main detection loop for real-time drone detection.
        Processes video frames, performs detection, and displays results.
        """
        if not self.load_model():
            return

        if not self.initialize_camera():
            return

        print("Starting drone detection... Press 'q' to quit.")

        while True:
            # Read frame from video source
            ret, frame = self.cap.read()

            if not ret:
                print("Error: Failed to read frame.")
                break

            # Perform detection and annotation
            annotated_frame, detected = self.detect_and_annotate(frame)

            # Trigger alert if drone detected
            if detected and not self.alert_triggered and self.alert_sound:
                self.alert_triggered = True
                threading.Thread(target=self.play_alert, daemon=True).start()

            # Display the annotated frame
            cv2.imshow("Drone Detection System", annotated_frame)

            # Exit on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Cleanup resources
        self.cleanup()

    def cleanup(self):
        """
        Release video capture and close all OpenCV windows.
        """
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        print("Detection stopped. Resources released.")


def main():
    """
    Main function to parse arguments and run the drone detection system.
    """
    parser = argparse.ArgumentParser(description="Drone Detection System using YOLO11")

    parser.add_argument(
        '--model',
        type=str,
        default='weights/best.pt',
        help='Path to YOLO11 model weights (default: weights/best.pt)'
    )

    parser.add_argument(
        '--source',
        type=str,
        default='0',
        help='Video source: 0 for webcam or path to video file (default: 0)'
    )

    parser.add_argument(
        '--conf',
        type=float,
        default=0.6,
        help='Confidence threshold for detection (default: 0.6)'
    )

    parser.add_argument(
        '--alert',
        type=str,
        default=None,
        help='Path to alert sound file (optional)'
    )

    args = parser.parse_args()

    # Convert source to int if it's a digit
    video_source = int(args.source) if args.source.isdigit() else args.source

    # Initialize and run detector
    detector = DroneDetector(
        model_path=args.model,
        alert_sound=args.alert,
        confidence_threshold=args.conf,
        video_source=video_source
    )

    detector.run()


if __name__ == "__main__":
    main()
