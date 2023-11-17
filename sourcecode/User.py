from FaceRecognition import SIFT, CNN, PretrainedModel
import cv2
import os
import re
from Pickling import PickleHelper
import numpy as np
from SpeakerRecognition import SpeakerRecognition

def countFilesInDir(dir : str) -> int:
    # counts the number of files in a dir
    count = 0
    for path in os.scandir(dir):
        if path.is_file():
            count += 1
    return count

def delete_pkl_files(directory):
    # deletes all the pickled features in the database in preparation for re-computing all of them
    # in case new feature extractor is added in the future.
    for personName in os.listdir(directory):
        for pickle in os.listdir(f"{directory}/{personName}/features"):
            os.unlink(f"{directory}/{personName}/features/{pickle}")

def replace_illegal_chars(filename):
    # replace illegal chars in case user enters them with the name, replace with underscore
    # These are the characters that are not allowed in a filename
    illegal_chars = r'[<>:"/\\|?*]'
    return re.sub(illegal_chars, '_', filename)

class UserEnrollment:
    def __init__(self):
        self.faceImageOutputPath = "./database/face"
        self.audioImageOutputPath = "./database/voice"
        if not os.path.exists(self.faceImageOutputPath):
            os.mkdir(self.faceImageOutputPath)
        if not os.path.exists(self.audioImageOutputPath):
            os.mkdir(self.audioImageOutputPath)
        self.speakerRecognition = SpeakerRecognition()
        self.pickler = PickleHelper()
        self.SIFT = SIFT()
        self.CNN = CNN()
        self.PretrainedModel = PretrainedModel()

    def reEnrollDatabase(self):
        '''
        It's common that we might want to modify how we store extracted features in the db, so 
        this function allows us to recompute new extracted features based off of enroll's functionality
        and replace all of the old features in the db
        '''
        # clear all .pkl files from db
        delete_pkl_files(self.faceImageOutputPath)
        
        for personName in os.listdir(self.faceImageOutputPath):
            for imageName in os.listdir(f"{self.faceImageOutputPath}/{personName}/images"):
                imagePath = f"{self.faceImageOutputPath}/{personName}/images/{imageName}"
                image = cv2.imread(imagePath)

                features = self.extractFeaturesFromImage(image)
                _, sift_test_descriptors, _ = features['SIFT'][0], features['SIFT'][1], features['SIFT'][2]
                cnn_embeddings = features['CNN']
                vgg_embeddings = features['VGG']

                num = countFilesInDir(f"{self.faceImageOutputPath}/{personName}/features")
                featuresSavePath = f"{self.faceImageOutputPath}/{personName}/features"
                T = (personName, image, sift_test_descriptors, cnn_embeddings, vgg_embeddings)
                self.pickler.save_to(f"{featuresSavePath}/{num+1}.pkl", T)
        
        return 'Done'

    def extractFeaturesFromImage(self, image : np.ndarray) -> dict:
        '''
        Given an image, extract features using all feature extractors
        '''
        # SIFT
        sift_test_keypoints, sift_test_descriptors, extracted_face_image = self.SIFT.process_face(image)

        # CNN
        cnn_embeddings = self.CNN.process_face(image)

        # Pretrained models
        pretrained_model_embeddings = self.PretrainedModel.process_face(image)

        return {
            'SIFT': (sift_test_keypoints, sift_test_descriptors, extracted_face_image),
            'CNN': cnn_embeddings,
            'VGG': pretrained_model_embeddings
        }

    def hybridEnroll(self, name : str, video : np.ndarray, audio : np.ndarray) -> dict:
        '''
        Given the captured video stream, voice recording, and name, enroll this user into the database
        by extracting features for both video (some select images from the video and from the audio)

        Returns a dict containing a random frame that we extracted features from (original and the extracted features image) and a mel spectrogram image of the voice
        '''
        d = {}

        # TODO:

        return d



    def enroll(self, name : str, image : np.ndarray) -> dict:
        '''
        Given the captured image and name, enroll this user into the database by saving
        their name, image, and extracted features
        '''
        name = replace_illegal_chars(name).lower().replace(' ', '_')
        imageSavePath = f"{self.faceImageOutputPath}/{name}/images"
        if not os.path.exists(imageSavePath):
            os.makedirs(imageSavePath)

        # 2. Extract Features
        features = self.extractFeaturesFromImage(image)
        sift_test_keypoints, sift_test_descriptors, extracted_face_image = features['SIFT'][0], features['SIFT'][1], features['SIFT'][2]
        cnn_embeddings = features['CNN']
        vgg_embeddings = features['VGG']
        
        # 3. Save tuple to database folder
        # save original image
        num = countFilesInDir(imageSavePath) # get number of current images
        filename = f"{name}-{num+1}" # save new image with a number of current images + 1
        cv2.imwrite(f"{imageSavePath}/{filename}.png", image)
        print(f"INFO: Now {name} has {num+1} images and extracted features in database")

        # ensure the directory for extracted features exists
        featuresSavePath = f"{self.faceImageOutputPath}/{name}/features"
        if not os.path.exists(featuresSavePath):
            os.mkdir(featuresSavePath)

        # pickle/save extracted features
        T = (name, image, sift_test_descriptors, cnn_embeddings, vgg_embeddings)
        self.pickler.save_to(f"{featuresSavePath}/{num+1}.pkl", T)

        # return sift image if face found
        return_dict = {}
        if sift_test_keypoints is None and sift_test_descriptors is None:
            # face not detected, return original image 
            _, ret_image = cv2.imencode('.png', image)
            return_dict['SIFT_image'] = ret_image  
        else:
            # store test image with SIFT keypoints superimposed to send to frontend
            test_img_kp = cv2.drawKeypoints(extracted_face_image, sift_test_keypoints, None)
            _, test_img_kp_bytes = cv2.imencode('.png', test_img_kp)
            return_dict['SIFT_image'] = test_img_kp_bytes
        return return_dict