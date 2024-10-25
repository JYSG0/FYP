# import required libraries
import tensorflow as tf
import cv2
import numpy as np
import matplotlib.pyplot as plt

# define the class
class LaneDetection:
    '''
    This class helps to streamline the task of using the lane detection models trained in this project.
    Now modified to work with real-time camera feed.
    '''
    def __init__(self, model_path, input_shape=(256, 320)):
        self.model = tf.keras.models.load_model(model_path, compile=False)
        self.input_shape = input_shape
    
    # preprocess the frame
    def __preprocess_frame(self, frame):
        frame = cv2.resize(frame, self.input_shape)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert to RGB for consistency
        frame = frame / 255.0  # Normalize to 0-1 range
        return np.expand_dims(frame, axis=0)
    
    # detect lanes in a single frame
    def detect_lanes(self, frame):
        processed_frame = self.__preprocess_frame(frame)
        pred = self.model.predict(processed_frame)[0]
        pred = (pred > 0.5).astype('int').reshape(*self.input_shape)
        return pred
    
    # visualize output by overlaying the predicted mask
    def visualize_output(self, frame, pred):
        # Resize prediction to original frame size for overlay
        pred_resized = cv2.resize(pred, (frame.shape[1], frame.shape[0]))
        mask = (pred_resized * 255).astype(np.uint8)
        
        # Apply the mask to the frame
        overlay = cv2.addWeighted(frame, 0.8, cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR), 0.2, 0)
        cv2.imshow('Lane Detection', overlay)

if __name__ == '__main__':
    # Initialize the model
    ld = LaneDetection('lane-detection-model.keras')
    
    # Open video capture (0 for default camera)
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open video stream.")
    
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Detect lane markings
            pred = ld.detect_lanes(frame)
            
            # Visualize the outputs
            ld.visualize_output(frame, pred)
            
            # Press 'q' to exit the loop
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
