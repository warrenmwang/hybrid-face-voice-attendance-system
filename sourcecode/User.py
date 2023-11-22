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
        self.audioOutputPath = "./database/voice"
        if not os.path.exists(self.faceImageOutputPath):
            os.mkdir(self.faceImageOutputPath)
        if not os.path.exists(self.audioOutputPath):
            os.mkdir(self.audioOutputPath)
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
        # TODO: add voice reenrollment!

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
        sift_test_keypoints, sift_test_descriptors, extracted_face_image = self.SIFT.process(image)

        # CNN
        cnn_embeddings = self.CNN.process(image)

        # Pretrained models
        pretrained_model_embeddings = self.PretrainedModel.process(image)

        return {
            'SIFT': (sift_test_keypoints, sift_test_descriptors, extracted_face_image),
            'CNN': cnn_embeddings,
            'VGG': pretrained_model_embeddings
        }

    def hybridEnroll(self, name : str, video : np.ndarray, audio : np.ndarray) -> dict:
        '''
        Given the captured video stream, voice recording, and name, enroll this user into the database
        by extracting features for both video (some select images from the video and from the audio)

        Inputs:
            name - str (e.g. bob)
            video - np.ndarray (shape (Frames, H, W, Dims)) -- e.g. (118, 480, 640, 3)
                if Dims == 3, RGB, else if Dims == 1, Grayscale [0,255] (NOTE: should be 3 here)
            audio - np.ndarray (shape (length, )) -- e.g. (375696,) -- array of ints of magnitudes 

        Returns a dict containing a random frame that we extracted features from (original and the extracted features image) and a mel spectrogram image of the voice
        '''
<<<<<<< Updated upstream
        d = {}

        # randomly select 30 frames from video to enroll
        rng = np.random.default_rng()
        random_frame_indices = rng.choice(len(video), size=30, replace=False)
        for i in random_frame_indices:
            tmp = self.enroll(name, video[i])
        d['SIFT_image'] = tmp['SIFT_image'] # grab the last sift image among the randomly selected frames to enroll

        # extract features from voice and enroll that
        mel_spectrogram = self.enrollVoice(name, audio)
        d['audio_melspectrogram'] = mel_spectrogram
=======
        # TODO:
>>>>>>> Stashed changes

        T = ()

        return T

    def enrollVoice(self, name, audio : np.ndarray):
        '''
        enroll a voice sample

        Inputs:
            audio - np.ndarray (shape (length, )) -- e.g. (375696,) -- array of ints of magnitudes 
        '''
        # 1. create dirs for audio and extracted features for this person if not exist
        name = replace_illegal_chars(name).lower().replace(' ', '_')
        audioSavePath = f"{self.audioOutputPath}/{name}/audio"
        featuresSavePath = f"{self.audioOutputPath}/{name}/features"
        if not os.path.exists(audioSavePath):
            os.makedirs(audioSavePath)
        if not os.path.exists(featuresSavePath):
            os.makedirs(featuresSavePath)

        # save original audio
        num = countFilesInDir(audioSavePath) # get number of current images
        filename = f"{name}-{num+1}" # save new image with a number of current images + 1
        self.pickler.save_to(f"{audioSavePath}/{filename}.pkl", audio)

        # 2. extract features
        mel_spectrogram = self.speakerRecognition.create_melspectrogram(audio)
        # TODO: modify the mel_spectrogram matrix of values into an actual "image"

        embeddings = self.PretrainedModel.process(mel_spectrogram)

        # 3. save extracted features
        T = (name, embeddings)
        self.pickler.save_to(f"{featuresSavePath}/{filename}.pkl", T)

        return mel_spectrogram

    def enroll(self, name : str, image : np.ndarray) -> np.ndarray:
        '''
        Given the captured image and name, enroll this user into the database by saving
        their name, image, and extracted features

        Returns the SIFt'ed image
        '''
        # 1. create dir for this person if not exist
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
        # TODO: remove unnecessary the image being pickled
        T = (name, image, sift_test_descriptors, cnn_embeddings, vgg_embeddings)
        self.pickler.save_to(f"{featuresSavePath}/{num+1}.pkl", T)

        # return sift image if face found
        ret_img = None
        if sift_test_keypoints is None and sift_test_descriptors is None:
            # face not detected, return original image 
            _, orig_img = cv2.imencode('.png', image)
            ret_img = orig_img
        else:
            # store test image with SIFT keypoints superimposed to send to frontend
            test_img_kp = cv2.drawKeypoints(extracted_face_image, sift_test_keypoints, None)
            _, test_img_kp_bytes = cv2.imencode('.png', test_img_kp)
            ret_img = test_img_kp_bytes
        return ret_img