import cv2
import os
import dlib
import numpy as np
from imutils import face_utils

def capture_face():
    video_capture = cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')
    
    while True:
        ret, frame = video_capture.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        # Draw rectangles around detected faces
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        cv2.imshow('Face Capture', frame)
        
        # Wait for a key press
        key = cv2.waitKey(1)
        
        # Press 'q' to quit the face capture
        if key == ord('q'):
            break
        
        # Press 's' to save the captured face
        if key == ord('s') and len(faces) > 0:
            # Save the captured face as 'user_face.jpg'
            cv2.imwrite('./user_face.jpg', frame)
            break
    
    video_capture.release()
    cv2.destroyAllWindows()

def find_matching_faces(input_face_path, collection_path):
    input_image = cv2.imread(input_face_path)
    gray_input = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
    
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
    facerec = dlib.face_recognition_model_v1('dlib_face_recognition_resnet_model_v1.dat')
    
    input_faces = detector(gray_input)
    
    if len(input_faces) == 0:
        print("No face detected in the input image.")
        return
    
    for input_face in input_faces:
        landmarks = predictor(gray_input, input_face)
        input_descriptor = np.array(facerec.compute_face_descriptor(input_image, landmarks))
        
        for filename in os.listdir(collection_path):
            if filename.endswith('.jpg') or filename.endswith('.png') or filename.endswith('.jpeg') or filename.endswith('.JPG') or filename.endswith('.JPEG'):
                image_path = os.path.join(collection_path, filename)
                collection_image = cv2.imread(image_path)
                
                if collection_image is None:
                    print(f"Failed to load image: {image_path}")
                    continue
                
                gray_collection = cv2.cvtColor(collection_image, cv2.COLOR_BGR2GRAY)
                collection_faces = detector(gray_collection)
                
                for collection_face in collection_faces:
                    collection_landmarks = predictor(gray_collection, collection_face)
                    collection_descriptor = np.array(facerec.compute_face_descriptor(collection_image, collection_landmarks))
                    
                    distance = np.linalg.norm(input_descriptor - collection_descriptor)
                    
                    # Set a threshold value for face matching
                    if distance < 0.5:  # originally 0.6
                        (x, y, w, h) = face_utils.rect_to_bb(collection_face)
                        cv2.rectangle(collection_image, (x, y), (x+w, y+h), (0, 255, 0), 2)
                
                        image_name = os.path.splitext(filename)[0]
                        save_path = os.path.join('./similar', f'{image_name}_similar.jpg')
                        cv2.imwrite(save_path, collection_image)
                
                cv2.imshow('Collection Image', collection_image)
                cv2.waitKey(1)
    
    cv2.destroyAllWindows()

if __name__ == '__main__':
    capture_face()
    find_matching_faces('./user_face.jpg', './Dataset')
