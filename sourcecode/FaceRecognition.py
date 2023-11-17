import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # turn off ts warnings
import cv2
import numpy as np
import keras
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense
from scipy.spatial.distance import cosine
from tensorflow.keras.applications.vgg19 import VGG19
from PIL import Image

class SIFT:
    def __init__(self):
        pass

    # detect face using cv2
    def detect_face(self, image):
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        return faces

    # return just face segment
    def segment_face(self, image, faces):
        # NOTE: wrapped in try/except because will throw an error if no faces are designated by the detect_face function
        try:
            if faces is not None and len(faces.shape) == 2:
                for (x, y, w, h) in faces:
                    face_image = image[y:y+h, x:x+w]
                    return face_image
            else:
                return None
        except:
            return None

    # SIFT feature extractor
    def extract_face_features(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        sift = cv2.SIFT_create()
        keypoints, descriptors = sift.detectAndCompute(gray, None)
        return keypoints, descriptors

    # use SIFT features to match face and compute similarity score
    def match_faces(self, test_descriptors, db_descriptors):
        bf = cv2.BFMatcher()
        # some images might be saved with NULL test_descriptors...
        if test_descriptors is None:
            print("INFO: test_descriptor is None")
            return 0.0
        if db_descriptors is None:
            print("INFO: db_descriptor is None")
            return 0.0

        matches = bf.match(test_descriptors, db_descriptors)
        distances = [m.distance for m in matches]
        score = sum(1 / (distance + 1e-6) for distance in distances)
        return score
    
    def process_face(self, test_image : np.ndarray):
        '''
        takes a test_img (cv2 image)
        and uses SIFT to get the features
        
        returns
            test_keypoints, test_descriptions, face_image
        '''
        # Detect the face in the test image.
        faces = self.detect_face(test_image)

        # Segment the face from the test image.
        face_image = self.segment_face(test_image, faces)
        if face_image is None:
            print("ERROR: No face detected in the test image or invalid face data.")
            return None, None, face_image

        # Extract features from the test face image.
        test_keypoints, test_descriptors = self.extract_face_features(face_image)

        return test_keypoints, test_descriptors, face_image

class CNN:
    def __init__(self):
        self.model = self.create_feature_extractor()

    # Creates a Simple CNN Model
    def create_feature_extractor(self, input_shape=(64, 64, 1), embedding_dim=128) -> Model:
        input = Input(shape=input_shape, name='input')
        x = Conv2D(32, (3, 3), activation='relu', padding='same', name='conv1')(input)
        x = MaxPooling2D((2, 2), name='pool1')(x)
        x = Conv2D(64, (3, 3), activation='relu', padding='same', name='conv2')(x)
        x = MaxPooling2D((2, 2), name='pool2')(x)
        x = Conv2D(128, (3, 3), activation='relu', padding='same', name='conv3')(x)
        x = MaxPooling2D((2, 2), name='pool3')(x)
        x = Flatten(name='flatten')(x)
        output = Dense(embedding_dim, activation='relu', name='embedding')(x)
        model = Model(inputs=input, outputs=output, name='feature_extractor')
        return model

    # Loads and Preprocesses an Image
    def preprocess_image(self, img : np.ndarray):
        # img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (64, 64))
        img = img.astype('float32') / 255.0
        img = np.expand_dims(img, axis=[0, -1])
        return img

    def process_face(self, test_image : np.ndarray):
        '''
        extract the features from the test_image and return the embeddings
        '''
        image = self.preprocess_image(test_image)
        embeddings = self.model.predict(image)
        return embeddings

    def compute_similarity_for_attendance(self, db_embeddings, test_image_embeddings) -> float:
        '''
        given the database image embedding and test image embedding, compute similarity between them
        '''
        # flatten embeddings to 1D for similarity computation
        test_image_embeddings = test_image_embeddings.flatten()
        db_image_embeddings = db_embeddings.flatten()
        # compute similarity using cosine
        similarity = 1 - cosine(test_image_embeddings, db_image_embeddings)
        return similarity

class PretrainedModel:
    def __init__(self):
        base_model = VGG19(weights='imagenet', include_top=False, pooling='avg')
        self.model = Model(inputs=base_model.input, outputs=base_model.output)

    def preprocess_image(self, image_array : np.ndarray):
        # Ensure the image is of the size 224x224
        img = Image.fromarray(image_array).resize((224, 224))
        img_array = np.array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = keras.applications.vgg16.preprocess_input(img_array)
        return img_array
    
    def process_face(self, test_image : np.ndarray):
        '''
        process the given input image and return the embeddings after passing the image thru the model
        '''
        img = self.preprocess_image(test_image)
        embeddings = self.model.predict(img)
        return embeddings

    def compute_similarity_for_attendance(self, db_embeddings, test_image_embeddings) ->float:
        '''
        given the database image embedding and test image embedding, compute similarity between them
        '''
        # flatten embeddings to 1D for similarity computation
        test_image_embeddings = test_image_embeddings.flatten()
        db_image_embeddings = db_embeddings.flatten()
        # compute similarity using cosine
        similarity = 1 - cosine(test_image_embeddings, db_image_embeddings)
        return similarity

