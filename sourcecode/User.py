from FaceRecognition import SIFT, CNN, PretrainedModel
import cv2
import os
import re
from Pickling import PickleHelper
import numpy as np
from SpeakerRecognition import SpeakerRecognition

def countFilesInDir(dir : str) -> int:
    # NOTE: running this over and over, with a large db, will be slow O(n). but since my db is smol, doesn't matter.
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
        # images (face)
        # clear all .pkl files from db
        delete_pkl_files(self.faceImageOutputPath)
        for personName in os.listdir(self.faceImageOutputPath):
            for imageName in os.listdir(f"{self.faceImageOutputPath}/{personName}/images"):
                # get image
                imagePath = f"{self.faceImageOutputPath}/{personName}/images/{imageName}"
                image = cv2.imread(imagePath)

                # extract features
                features = self.extractFeaturesFromImage(image)
                _, sift_test_descriptors, _ = features['SIFT'][0], features['SIFT'][1], features['SIFT'][2]
                cnn_embeddings = features['CNN']
                vgg_embeddings = features['VGG']

                # save features
                featuresSavePath = f"{self.faceImageOutputPath}/{personName}/features"
                num = countFilesInDir(featuresSavePath)
                T = (personName, image, sift_test_descriptors, cnn_embeddings, vgg_embeddings)
                self.pickler.save_to(f"{featuresSavePath}/{num+1}.pkl", T)
        
        # audio (voice)
        # clear all .pkl files from db
        delete_pkl_files(self.audioOutputPath)
        for personName in os.listdir(self.audioOutputPath):
            for audioName in os.listdir(f"{self.audioOutputPath}/{personName}/audio"):
                # get audio
                audioPath = f"{self.audioOutputPath}/{personName}/audio/{audioName}"
                audio = self.speakerRecognition.readAudioFileIntoNpNdArray(audioPath)

                # extract features
                _, mel_spectrogram_img = self.speakerRecognition.convertAudioToImageRepresentation(audio)
                embeddings = self.PretrainedModel.process(mel_spectrogram_img)

                # save features
                featuresSavePath = f"{self.audioOutputPath}/{personName}/features"
                num = countFilesInDir(featuresSavePath)
                filename = f"{personName}-{num+1}"
                T = (personName, embeddings)
                self.pickler.save_to(f"{featuresSavePath}/{filename}.pkl", T)
        
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

    def hybridEnroll(self, name : str, video : np.ndarray, audio : np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        '''
        Given the captured video stream, voice recording, and name, enroll this user into the database
        by extracting features for both video (some select images from the video and from the audio)

        Inputs:
            name - str (e.g. bob)
            video - np.ndarray (shape (Frames, H, W, Dims)) -- e.g. (118, 480, 640, 3)
                if Dims == 3, BGR, else if Dims == 1, Grayscale [0,255] (NOTE: should be 3 here)
            audio - np.ndarray (shape (length, )) -- e.g. (375696,) -- array of ints of magnitudes 

        Returns a dict containing a random frame that we extracted features from (original and the extracted features image) and a mel spectrogram image of the voice
        '''
        N = 10

        # randomly select N frames from video to enroll
        rng = np.random.default_rng()
        random_frame_indices = rng.choice(len(video), size=N, replace=False)
        orig_image, sift_image = None, None
        for i in random_frame_indices:
            sift_image = self.enrollImage(name, video[i])
            orig_image = video[i]

        # extract features from voice and enroll that
        mel_spectrogram_img = self.enrollVoice(name, audio)

        # sift image is already cv2 encoded bytes, now do that for orig_image and mel_spectrogram
        _, orig_image = cv2.imencode('.png', orig_image)
        _, mel_spectrogram_img = cv2.imencode('.png', mel_spectrogram_img)

        print(f"DEBUG: {orig_image.shape=}")
        print(f"DEBUG: {sift_image.shape=}")
        print(f"DEBUG: {mel_spectrogram_img.shape=}")

        T = (orig_image, sift_image, mel_spectrogram_img)
        return T

    def enrollVoice(self, name : str , audio : np.ndarray) -> np.ndarray:
        '''
        enroll a voice sample

        Inputs:
            audio (np.ndarray): (shape (length, )) -- e.g. (375696,) -- array of ints of magnitudes 
        Outputs:
            mel_spectrogram_img (np.ndarray):  the mel spectrogram representation of the audio time series array as a uint8 image
                expect this to be RGB uint8? 
        '''
        # 1. create dirs for audio and extracted features for this person if not exist
        name = replace_illegal_chars(name).lower().replace(' ', '_')
        audioSavePath = f"{self.audioOutputPath}/{name}/audio"
        featuresSavePath = f"{self.audioOutputPath}/{name}/features"
        if not os.path.exists(audioSavePath):
            os.makedirs(audioSavePath)
        if not os.path.exists(featuresSavePath):
            os.makedirs(featuresSavePath)
        
        # data processing: convert into float64 dtype and normalize values into [-1.0,1.0]
        audio = audio.astype(float) # defaults to float64
        audio = audio / np.max(np.abs(audio))
        print(f"DEBUG: {audio.min()=} {audio.max()=}")

        # save original audio
        num = countFilesInDir(audioSavePath) # get number of current images
        filename = f"{name}-{num+1}" # save new image with a number of current images + 1
        filepath = f"{audioSavePath}/{filename}.wav"
        self.speakerRecognition.saveAudioNpNdArrayAsFile(audio, filepath)

        # 2. extract features
        _, mel_spectrogram_img = self.speakerRecognition.convertAudioToImageRepresentation(audio)
        embeddings = self.PretrainedModel.process(mel_spectrogram_img)

        # 3. save extracted features
        T = (name, embeddings)
        self.pickler.save_to(f"{featuresSavePath}/{filename}.pkl", T)

        return mel_spectrogram_img

    def enrollImage(self, name : str, image : np.ndarray) -> np.ndarray:
        '''
        Given the captured image and name, enroll this user into the database by saving
        their name, image, and extracted features

        Inputs:
            name - str (e.g. bob)
            image - np.ndarray (e.g. (H, W, dims))
                assume image is BGR
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