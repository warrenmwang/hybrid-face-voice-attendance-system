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
        self.pickler = PickleHelper()
        self.SIFT = SIFT()
        self.CNN = CNN()
        self.PretrainedModel = PretrainedModel()

    def record(self, attendance_file : str, test_image : np.ndarray, featureExtractionMethod : str, scoreThreshold : float) -> tuple[str, float]:
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
                name, db_image, db_test_descriptors = T[0], T[1], T[2]
                # images that SIFT failed to grab the features will be of Nonetype, skip those by setting score to 0
                if type(db_test_descriptors) == type(None):
                    score = 0
                else:
                    score = self.SIFT.match_faces(test_descriptors, db_test_descriptors)
                scores.append((name, score, db_image))
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
                name, db_image, _, db_image_embeddings = T[0], T[1], T[2], T[3]
                score = self.CNN.compute_similarity_for_attendance(db_image_embeddings, test_image_embeddings)
                scores.append((name, score, db_image))
        return scores


    def getScoresViaPretrainedModel(self, test_image : np.ndarray):
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
                name, db_image, _, _, db_image_embeddings = T[0], T[1], T[2], T[3], T[4]
                score = self.PretrainedModel.compute_similarity_for_attendance(db_image_embeddings, test_image_embeddings)
                scores.append((name, score, db_image))
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

    def start(self, scoreThreshold_SIFT : float, scoreThreshold_CNN : float, scoreThreshold_VGG16 : float, groundTruth : str, numItersPerMethod : int) -> None:
        '''Clears any past variables and restarts a new session with new inputs'''
        self.scoreThreshold_SIFT = scoreThreshold_SIFT
        self.scoreThreshold_CNN = scoreThreshold_CNN
        self.scoreThreshold_VGG16 = scoreThreshold_VGG16
        self.groundTruth = groundTruth
        self.numItersPerMethod = numItersPerMethod
        self.featureExtractionMethods = ['SIFT', 'CNN', 'VGG16']
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
        elif featureExtractionMethod == "VGG16":
            scores = self.getScoresViaPretrainedModel(image)
            scoreThreshold = self.scoreThreshold_VGG16
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
            
            ret_str = ""
            ret_str += "<h3>Number of Takes till Correct Match:</h3>"
            ret_str += "<p>If count is equal to the number of iterations to run for each method, then never matched within given number of iterations.</p>"
            for m, c in CountsTillCorrect:
                ret_str += f"{m}: {c}<br>"

            ret_str += "<h3>TARS:</h3>"
            for m,p in TARs:
                ret_str += f"{m}: {p}<br>"

            ret_str += "<h3>FARs:</h3>"
            for m,p in FARs:
                ret_str += f"{m}: {p}<br>"

            ret_str += "<h3>FRRs:</h3>"
            for m,p in FRRs:
                ret_str += f"{m}: {p}<br>"

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

    def enrollImage(self, name, picture) -> np.ndarray:
        '''
        regular enrollment uses just a picture of the user to enroll
        returns the image with the face detected and SIFT keypoints marked
        '''
        return self.user_enrollment.enrollImage(name, picture)

    def recordAttendance(self, image : np.ndarray, featureExtractionMethod : str, scoreThreshold : float) -> str:
        return self.attendance_recording.record(self.attendance_file, image, featureExtractionMethod, scoreThreshold)

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

    def automatedAttendanceTestingStart(self, scoreThreshold_SIFT : float, scoreThreshold_CNN : float, scoreThreshold_VGG16 : float, groundTruth : str, numItersPerMethod : int) -> None:
        self.automated_attendance_testing.start(scoreThreshold_SIFT, scoreThreshold_CNN, scoreThreshold_VGG16, groundTruth, numItersPerMethod)
    
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
