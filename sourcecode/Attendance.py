import datetime
import csv
import os
from User import UserEnrollment
from Pickling import PickleHelper
from FaceRecognition import SIFT, CNN, PretrainedModel
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from io import BytesIO
from SpeakerRecognition import SpeakerRecognition

# toggles for running as webserver
plt.ioff() # turn off matplotlib interactive mode
matplotlib.use('Agg') # set matplotlib to use non-GUI backend

def touch(path):
    with open(path, 'a'):
        os.utime(path, None)

class AttendanceRecording:
    def __init__(self, databasePath : str):
        self.attendance_allowed_time_interal = 3600 # [seconds] (1 hour)
        self.recently_recognized_students = {} # username -> logged in time (datetime)
        self.databasePath = databasePath
        self.faceDatabasePath = f"{databasePath}/face"
        self.voiceDatabasePath = f"{databasePath}/voice"
        self.pickler = PickleHelper()
        self.SIFT = SIFT()
        self.CNN = CNN()
        self.PretrainedModel = PretrainedModel()
        self.speakerRecognition = SpeakerRecognition()


    def recordViaHybrid(self, attendance_file : str, video : np.ndarray, audio : np.ndarray, scoreThreshold : float) -> tuple[str, float]:
        '''
        Similar to recordViaFace, will try to identify a user's presence in the video, authenticates with voice+face.

        TODO: For now, hybrid identification system is using a simply mean of the two individual match scores?
        '''
        # get scores for face

        # grab N frames from video, for each frame get top scorer 
        # then find the top scorer of N top scorers
        N = 5
        rng = np.random.default_rng()
        random_frame_indices = rng.choice(len(video), size=N, replace=False)
        face_scores = []
        for i in random_frame_indices:
            test_image = video[i]
            face_score = self.getScoresViaPretrainedModel(test_image)
            face_score.sort(key=lambda x: x[1], reverse=True)
            top_scorer = face_score[0]
            face_scores.append(top_scorer)
        face_scores.sort(key=lambda x: x[1], reverse=True) 
        top_face_scorer = face_scores[0]
        print(f"DEBUG: {top_face_scorer=}")
        
        # get score for voice
        voice_scores = self.getScoresViaVoice(audio)
        print(f"DEBUG: {voice_scores=}")

        voice_scores.sort(key=lambda x: x[1], reverse=True)
        top_voice_scorer = voice_scores[0]
        print(f"DEBUG: {top_voice_scorer=}")

        # combine face and voice scores
        # TODO: How to do hybrid identification with two data modalities? image+audio??
        # just combine the embedding vectors and do cosine similarity?
        # scores = []
        # FIXME: temporary setting -- use regular face for now...
        top_scorer = top_face_scorer

        name, raw_score = top_scorer[0], top_scorer[1]
        if raw_score > scoreThreshold:
            username = name

        if username is None:
            print(f"INFO: Top scorer doesn't pass confidence threshold")
            return None, None
        if self.take_attendance_for_username(username, attendance_file):
            return username, raw_score
        print(f"INFO: {username} has recently already been logged for attendance.")
        return None, None

    def recordViaFace(self, attendance_file : str, test_image : np.ndarray, featureExtractionMethod : str, scoreThreshold : float) -> tuple[str, float]:
        '''
        Try to recognize student from given image object using the specified feature extraction method
        If recognizes a student and has taken attendance, return that student's name and the raw similarity score computed
        otherwise returns None

        Students should implement the following:
        1. Capture images continuously unless interrupted.
        2. Translate image into features.
        3. Match features with database and record attendance.
        '''
        # init username as None, will stay none if no student score is high enough
        username = None 

        if featureExtractionMethod == "SIFT":
            scores = self.getScoresViaSIFT(test_image)
        elif featureExtractionMethod == "CNN":
            scores = self.getScoresViaCNN(test_image)
        elif featureExtractionMethod == "PretrainedModel":
            scores = self.getScoresViaPretrainedModel(test_image)
        else:
            print('ERROR: Unknown feature extraction method.')
            return

        # Sort the scores in descending order (best match first).
        scores.sort(key=lambda x: x[1], reverse=True)
        
        # check top score
        top_scorer = scores[0]
        name, raw_score, _ = top_scorer[0], top_scorer[1], top_scorer[2]
        if raw_score > scoreThreshold:
            username = name

        if username is None:
            print(f"INFO: Top scorer doesn't pass confidence threshold")
            return None, None
        if self.take_attendance_for_username(username, attendance_file):
            return username, raw_score
        return None, None

    def getScoresViaSIFT(self, test_image : np.ndarray):
        '''
        Get scores via SIFT method

        similarity scores (range [0, 1])
        (name, image, sift_save_object, cnn_save_object, pretrained_save_object)
        '''
        _, test_descriptors, _ = self.SIFT.process(test_image)
        scores = [] 
        for person in os.listdir(self.faceDatabasePath):
            for pickle in os.listdir(f"{self.faceDatabasePath}/{person}/features"):
                picklePath = f"{self.faceDatabasePath}/{person}/features/{pickle}"
                T = self.pickler.load_back(picklePath) 
                # name, db_image, db_test_descriptors = T[0], T[1], T[2]
                name, db_test_descriptors = T[0], T[2]
                # images that SIFT failed to grab the features will be of Nonetype, skip those by setting score to 0
                if type(db_test_descriptors) == type(None):
                    score = 0
                else:
                    score = self.SIFT.match_faces(test_descriptors, db_test_descriptors)
                # scores.append((name, score, db_image))
                scores.append((name, score))
        return scores

    def getScoresViaCNN(self, test_image : np.ndarray):
        '''
        Get scores via CNN method

        similarity scores (range [0, 1])
        (name, image, sift_save_object, cnn_save_object, pretrained_save_object)
        '''
        scores = []
        test_image_embeddings = self.CNN.process(test_image)
        for person in os.listdir(self.faceDatabasePath):
            for pickle in os.listdir(f"{self.faceDatabasePath}/{person}/features"):
                picklePath = f"{self.faceDatabasePath}/{person}/features/{pickle}"
                T = self.pickler.load_back(picklePath) 
                # name, db_image, _, db_image_embeddings = T[0], T[1], T[2], T[3]
                name, db_image_embeddings = T[0], T[3]
                score = self.CNN.compute_similarity_for_attendance(db_image_embeddings, test_image_embeddings)
                # scores.append((name, score, db_image))
                scores.append((name, score))
        return scores

    def getScoresViaPretrainedModel(self, test_image : np.ndarray) -> list[tuple[str, float, np.ndarray]]:
        '''
        Get scores via Pretrained Model method

        similarity scores (range [0, 1])
        (name, image, sift_save_object, cnn_save_object, pretrained_save_object)
        '''
        scores = []
        test_image_embeddings = self.PretrainedModel.process(test_image)
        for person in os.listdir(self.faceDatabasePath):
            for pickle in os.listdir(f"{self.faceDatabasePath}/{person}/features"):
                picklePath = f"{self.faceDatabasePath}/{person}/features/{pickle}"
                T = self.pickler.load_back(picklePath) 
                # name, db_image, _, _, db_image_embeddings = T[0], T[1], T[2], T[3], T[4]
                name, db_image_embeddings = T[0], T[4]
                score = self.PretrainedModel.compute_similarity_for_attendance(db_image_embeddings, test_image_embeddings)
                # scores.append((name, score, db_image))
                scores.append((name, score))
        return scores
    
    def getScoresViaVoice(self, testAudio : np.ndarray) -> list[tuple[str,float]]:
        '''
        compute match scores for testAudio against all the audio samples for each person in the db

        Returns:
            a list of tuples of (name, score) for each person's voice audio sample and stuff and the match score via mel spectrogram
        '''
        scores = []
        _, mel_spectrogram_img = self.speakerRecognition.convertAudioToImageRepresentation(testAudio)
        test_mel_spectrogram_img_embeddings = self.PretrainedModel.process(mel_spectrogram_img)
        for person in os.listdir(self.voiceDatabasePath):
            for pickle in os.listdir(f"{self.voiceDatabasePath}/{person}/features"):
                picklePath = f"{self.voiceDatabasePath}/{person}/features/{pickle}"
                T = self.pickler.load_back(picklePath)
                db_name, db_embeddings = T[0], T[1]
                score = self.PretrainedModel.compute_similarity_for_attendance(db_embeddings, test_mel_spectrogram_img_embeddings)
                scores.append((db_name, score))
        return scores

    def take_attendance_for_username(self, username : str, attendance_file : str) -> bool:
        '''
        tries to take attendance for the given username, handles whether they have been already marked present or not recently
        returns True if did take attendance, otherwise return False
        '''
        # if username had already been recently "logged in", don't record attendance again
        if username in self.recently_recognized_students.keys():
            # check if it's time to release student from list if past time interval
            # if not, skip
            # if past, remove them, then log in 
            curr_time = datetime.datetime.now()
            delta = curr_time - self.recently_recognized_students[username]
            sec = delta.total_seconds()
            if sec > self.attendance_allowed_time_interal:
                del self.recently_recognized_students[username]
                print(f"INFO: Logging attendance for {username}")
                recorded_time = self.update_attendance_file(username, attendance_file)
                self.recently_recognized_students[username] = recorded_time
                return True
            return False

        # username has not been recently logged in, log them in now
        print(f"INFO: Logging attendance for {username}")
        recorded_time = self.update_attendance_file(username, attendance_file)
        self.recently_recognized_students[username] = recorded_time
        return True

    def update_attendance_file(self, username : str, attendance_file : str) -> datetime.datetime:
        '''
        Record attendance for the given username
        '''
        current_time = datetime.datetime.now()
        current_time_str = current_time.strftime("%Y-%m-%d %H:%M:%S")
        with open(attendance_file, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([username, current_time_str])
        return current_time
    
    def reset_attendance_recording(self):
        # resets current attendance recording session so that we can mark same students again
        self.recently_recognized_students = {}

class AutomatedAttendanceTesting(AttendanceRecording):
    # inherit AttendanceRecording to get scores using same methods
    def __init__(self, databasePath : str):
        super().__init__(databasePath)

    def start(self, scoreThreshold_SIFT : float, scoreThreshold_CNN : float, scoreThreshold_VGG : float, groundTruth : str, numItersPerMethod : int) -> None:
        '''Clears any past variables and restarts a new session with new inputs'''
        self.scoreThreshold_SIFT = scoreThreshold_SIFT
        self.scoreThreshold_CNN = scoreThreshold_CNN
        self.scoreThreshold_VGG = scoreThreshold_VGG
        self.groundTruth = groundTruth
        self.numItersPerMethod = numItersPerMethod
        self.featureExtractionMethods = ['SIFT', 'CNN', 'VGG']
        # self.countsPassThreshold = {k:0 for k in self.featureExtractionMethods}
        self.counts_correct = {k:0 for k in self.featureExtractionMethods}
        self.counts_incorrect = {k:0 for k in self.featureExtractionMethods}
        self.numberIterationsTillCorrectCount = {k:0 for k in self.featureExtractionMethods}
        self.numberIterationsTillCorrectFlag = {k:False for k in self.featureExtractionMethods}
        self.falseRejectCounts = {k:0 for k in self.featureExtractionMethods}
    
    def process(self, image : np.ndarray, featureExtractionMethod : str) -> tuple[str, float, float]:
        '''
        Returns the top scorer running the given featureExtractionMethod
        Saves whether this was a hit or not for this automated testing session using the ground truth information
        '''
        if featureExtractionMethod == "SIFT":
            scores = self.getScoresViaSIFT(image)
            scoreThreshold = self.scoreThreshold_SIFT
        elif featureExtractionMethod == "CNN":
            scores = self.getScoresViaCNN(image)
            scoreThreshold = self.scoreThreshold_CNN
        elif featureExtractionMethod == "VGG":
            scores = self.getScoresViaPretrainedModel(image)
            scoreThreshold = self.scoreThreshold_VGG
        else:
            print('ERROR: Unknown feature extraction method.')
            return
        
        # Sort the scores in descending order (best match first).
        scores.sort(key=lambda x: x[1], reverse=True)
        
        # check top score
        top_scorer = scores[0]
        name, raw_score, _ = top_scorer[0], top_scorer[1], top_scorer[2]

        # if pass threshold check to see if match with ground truth
        if raw_score > scoreThreshold:
            if not self.numberIterationsTillCorrectFlag[featureExtractionMethod]:
                self.numberIterationsTillCorrectCount[featureExtractionMethod] += 1

            if name == self.groundTruth:
                self.counts_correct[featureExtractionMethod] += 1
                self.numberIterationsTillCorrectFlag[featureExtractionMethod] = True
            else:
                self.counts_incorrect[featureExtractionMethod] += 1
        else:
            if not self.numberIterationsTillCorrectFlag[featureExtractionMethod]:
                self.numberIterationsTillCorrectCount[featureExtractionMethod] += 1
            # testing assumes that the ground truth subject is present in all images, so count if score below threshold as a false reject
            self.falseRejectCounts[featureExtractionMethod] += 1

        return name, raw_score, scoreThreshold


    def stop(self) -> str:
        # return results of this automated testing session as a str
        try:
            TARs = [(method, num_accepts/self.numItersPerMethod) for method, num_accepts in self.counts_correct.items()]
            FARs = [(method, num_accepts/self.numItersPerMethod) for method, num_accepts in self.counts_incorrect.items()]
            FRRs = [(method, rejects/self.numItersPerMethod) for method, rejects in self.falseRejectCounts.items()]
            # can't do TRR with current setup unless add another button to toggle a different mode of testing where subject in image is NOT the ground truth (extra) (reject if below threshold and ground truth is not equal to result)
            CountsTillCorrect = [(method, counts) for method, counts in self.numberIterationsTillCorrectCount.items()]
            
            # chars for formatting
            # tab_char = '&#x09;'
            tab_char = '    '

            ret_str = ""
            ret_str += "<h3>Number of Takes till Correct Match:</h3>"
            ret_str += "<p>If count is equal to the number of iterations to run for each method, then never matched within given number of iterations.</p>"
            for m, c in CountsTillCorrect:
                ret_str += f"{tab_char}{m}: {c}<br>"

            ret_str += "<h3>TARS:</h3>"
            for m,p in TARs:
                ret_str += f"{tab_char}{m}: {p}<br>"

            ret_str += "<h3>FARs:</h3>"
            for m,p in FARs:
                ret_str += f"{tab_char}{m}: {p}<br>"

            ret_str += "<h3>FRRs:</h3>"
            for m,p in FRRs:
                ret_str += f"{tab_char}{m}: {p}<br>"

            return ret_str
        except Exception as e:
            return f"An Error occurred ({e}), please try again."


class AttendanceSystem:
    def __init__(self):
        self.databasePath = './database'
        self.attendance_file = "attendance.csv"
        self.user_enrollment = UserEnrollment()
        self.attendance_recording = AttendanceRecording(self.databasePath)
        self.attendance_plotting = AttendancePlotting()
        self.automated_attendance_testing = AutomatedAttendanceTesting(self.databasePath)
        # make attendance file if not exists
        if not os.path.exists(self.attendance_file):
            touch(self.attendance_file)

    def hybridEnroll(self, name : str, video : np.ndarray, audio : np.ndarray) -> (np.ndarray, np.ndarray, np.ndarray):
        '''
        hybrid enrollment uses both a videostream of the user and an audiostream of the using saying a particular phrase
        returns tuple of (random orig frame, extracted feature frame, mel spectrogram of voice)
        '''
        return self.user_enrollment.hybridEnroll(name, video, audio)
    
    def recordAttenanceViaHybrid(self, video : np.ndarray, audio : np.ndarray, scoreThreshold : float) -> tuple[str, float]:
        '''
        only allow the use of the VGG feature extractor for both the video and the audio (mel_spectrogram) for now

        Returns 
            name (str): authenticated user
            score (float): match score
                both are None if not pass given threshold
        '''
        return self.attendance_recording.recordViaHybrid(self.attendance_file, video, audio, scoreThreshold)

    def enrollImage(self, name, picture) -> np.ndarray:
        '''
        regular enrollment uses just a picture of the user to enroll
        returns the image with the face detected and SIFT keypoints marked
        '''
        return self.user_enrollment.enrollImage(name, picture)

    def recordAttendanceViaFace(self, image : np.ndarray, featureExtractionMethod : str, scoreThreshold : float) -> tuple[str, float]:
        '''
        Tries to identify the person in the image with the given feature extraction method and a score threshold

        Returns 
            name (str): authenticated user
            score (float): match score
                both are None if not pass given threshold
        '''
        return self.attendance_recording.recordViaFace(self.attendance_file, image, featureExtractionMethod, scoreThreshold)

    def plotAttendance(self) -> BytesIO:
        return self.attendance_plotting.plot(self.attendance_file)

    def resetAttendance(self) -> None:
        # resets the current attendance session
        self.attendance_recording.reset_attendance_recording()

    def clearAttendanceFile(self) -> None:
        # overwrite an empty attendance.csv over the existing attendance.csv 
        with open(self.attendance_file, "w"):
            pass

    def reExtractAllImages(self) -> str:
        return self.user_enrollment.reEnrollDatabase()

    def automatedAttendanceTestingStart(self, scoreThreshold_SIFT : float, scoreThreshold_CNN : float, scoreThreshold_VGG : float, groundTruth : str, numItersPerMethod : int) -> None:
        self.automated_attendance_testing.start(scoreThreshold_SIFT, scoreThreshold_CNN, scoreThreshold_VGG, groundTruth, numItersPerMethod)
    
    def automatedAttendanceTestingProcess(self, image : np.ndarray, featureExtractionMethod : str) -> tuple[str, float, float]:
        return self.automated_attendance_testing.process(image, featureExtractionMethod)

    def automatedAttendanceTestingStop(self) -> str:
        return self.automated_attendance_testing.stop()


class AttendancePlotting:
    def plot(self, attendance_file : str) -> BytesIO:
        '''
        Returns the created figure image to be displayed on frontend
        Students should implement the following:
        1. Analyze attendance.csv.
        2. Plot the number of presence for each student.
        '''
        df = pd.read_csv(attendance_file, names=['user', 'timestamp'])

        # Count the occurrences of each user
        user_counts = df['user'].value_counts()

        # Create the histogram
        fig, ax = plt.subplots(figsize=(7,7))
        ax.bar(user_counts.index, user_counts.values)
        ax.set_xlabel('User')
        ax.set_ylabel('Count')
        ax.set_title('User Occurrences')

        # Save the figure into a BytesIO object and return it 
        buf = BytesIO()
        fig.savefig(buf, format='png')
        buf.seek(0)
        return buf
