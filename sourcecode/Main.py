from flask import Flask, render_template, request, send_file, jsonify
from werkzeug.utils import secure_filename
import cv2
import numpy as np
from io import BytesIO
from Attendance import AttendanceSystem
import os
from pydub import AudioSegment
from Pickling import PickleHelper
import base64

def ndarray_to_base64(image : np.ndarray) -> str:
    '''encode an image (in numpy ndarray format) into a b64 string'''
    raw_bytes = BytesIO(image.tobytes())
    return base64.b64encode(raw_bytes.read()).decode()

system = AttendanceSystem()

app = Flask(__name__, static_folder='static', static_url_path='')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/enrollment', methods=['POST'])
def upload():
    # get image, username
    uploaded_image = request.files['image']
    username = request.form.get('text')

    # Read the image using OpenCV, color format is BGR
    image = cv2.imdecode(np.frombuffer(uploaded_image.read(), np.uint8), cv2.IMREAD_COLOR)

    # enroll user, extract features, then send the extracted features to the screen (for now just show SIFT?)
    # returns the image as a np.ndarray
    img = system.enrollImage(username, image)
    SIFT_img_with_keypoints = BytesIO(img.tobytes())

    return send_file(SIFT_img_with_keypoints, mimetype='image/png')

@app.route('/enrollment/hybrid', methods=['POST'])
def uploadHybrid():
    # get video, audio, username
    videoAndAudio = request.files['videoAndAudio']
    username = request.form.get('text')

    videoAndAudio.save('temp.webm')

    # Use OpenCV to read the video frames into a numpy array
    video = cv2.VideoCapture("temp.webm")
    frames = []
    while True:
        ret, frame = video.read()
        if not ret:
            break
        frames.append(frame)
    video_frames = np.array(frames) # color format should be BGR bc using cv2
    
    # read the audio into a numpy array
    audio = AudioSegment.from_file("temp.webm", format="webm")
    audio_samples = np.array(audio.get_array_of_samples())
    
    os.remove("temp.webm")

    # TODO: return the [random orig frame, extracted feature frame, mel spectrogram of voice] (tuple)
    T = system.hybridEnroll(username, video_frames, audio_samples)
    # assume all images are numpy arrays.
    # convert images into base64 encoded strings to jsonify and then send to frontend
    face_img = ndarray_to_base64(T[0])
    features_extracted_img = ndarray_to_base64(T[1])
    audio_mel_spectrogram_img = ndarray_to_base64(T[2])

    return jsonify([face_img, features_extracted_img, audio_mel_spectrogram_img]), 200

@app.route('/attendance', methods=['POST'])
def attendance():
    '''
    Gets pictures from the continuous attendance video feed tries to take attendance
    If system identifies a student in the captured picture, return the name of the person, 
    the raw similarity score found with the person in the db, and the score threshold used to determine 
    this score is good enough to count
    '''
    image = request.files['image']
    image = cv2.imdecode(np.frombuffer(image.read(), np.uint8), cv2.IMREAD_COLOR)
    featureExtractionMethod = request.form.get('featureExtractionMethod')
    scoreThreshold = float(request.form.get('scoreThreshold'))
    name, raw_score = system.recordAttendanceViaFace(image, featureExtractionMethod, scoreThreshold)
    return jsonify({'name': name,
                    'raw_score': raw_score, 
                    'score_threshold': scoreThreshold}), 200

@app.route('/attendanceHybrid', methods=['POST'])
def attendanceHybrid():
    '''
    hybrid attendance video+audio identification
    '''
    # get video, audio, username
    videoAndAudio = request.files['videoAndAudio']
    # username = request.form.get('text')

    videoAndAudio.save('temp.webm')

    # Use OpenCV to read the video frames into a numpy array
    video = cv2.VideoCapture("temp.webm")
    frames = []
    while True:
        ret, frame = video.read()
        if not ret:
            break
        frames.append(frame)
    video_frames = np.array(frames) # color format should be BGR bc using cv2
    
    # read the audio into a numpy array
    audio = AudioSegment.from_file("temp.webm", format="webm")
    audio_samples = np.array(audio.get_array_of_samples())
    
    os.remove("temp.webm")

    scoreThreshold = 0.5 # TODO:
    name, raw_score = system.recordAttenanceViaHybrid(video_frames, audio_samples, scoreThreshold)
    print(f"DEBUG: {name=} {raw_score=}")
    return jsonify({'name': name,
                    'raw_score': raw_score, 
                    'score_threshold': scoreThreshold}), 200


@app.route('/reset_attendance', methods=['POST'])
def reset_attendance():
    '''
    resets the current attendance session so can take more attendance without restarting web server
    or waiting for the attendance interval for logging students again (default 1 hour)
    '''
    system.resetAttendance()
    return jsonify({'message': None}), 200

@app.route('/plot_attendance_histogram', methods=['GET'])
def plot_attendance():
    '''
    Gives images based on the analysis of the attendance.csv file contents
    '''
    img = system.plotAttendance()
    return send_file(img, mimetype='image/png')

@app.route('/clear_attendance_file', methods=['POST'])
def clear_attendance_file():
    '''
    Clears the contents of the attendance.csv file
    '''
    system.clearAttendanceFile()
    return jsonify({'message': None}), 200

@app.route('/re-extract-all-images', methods=['POST'])
def re_extract_all_images():
    '''
    Recomputes the feature extraction for all the images in the current database
    May take a little while if a lot of images present
    Will overwrite the current existing extracted features

    Originally written due to existence of some SIFT features being Nonetypes in the db
    '''
    x = system.reExtractAllImages()
    return jsonify({'message': x}), 200

@app.route('/automated_attendance_testing/start', methods=['POST'])
def automated_attendance_testing_start():
    '''
    Start of automated testing
    '''
    scoreThreshold_SIFT = float(request.form.get('scoreThreshold_SIFT'))
    scoreThreshold_CNN = float(request.form.get('scoreThreshold_CNN'))
    scoreThreshold_VGG = float(request.form.get('scoreThreshold_VGG'))
    groundTruth = request.form.get('groundTruth')
    numItersPerMethod = int(request.form.get('numItersPerMethod'))
    print(f"INFO: Starting Automated Attendance Testing with params: {scoreThreshold_CNN=} {scoreThreshold_SIFT=} {scoreThreshold_VGG=} {groundTruth=}")
    system.automatedAttendanceTestingStart(scoreThreshold_SIFT, scoreThreshold_CNN, scoreThreshold_VGG, groundTruth, numItersPerMethod)
    return jsonify({'message': None}), 200

@app.route('/automated_attendance_testing/process', methods=['POST'])
def automated_attendance_testing_process():
    '''
    Automates the process of testing and evaluating how good all the feature extraction methods are
    '''
    image = request.files['image']
    image = cv2.imdecode(np.frombuffer(image.read(), np.uint8), cv2.IMREAD_COLOR)
    featureExtractionMethod = request.form.get('featureExtractionMethod')
    name, raw_score, scoreThreshold = system.automatedAttendanceTestingProcess(image, featureExtractionMethod)
    return jsonify({'name': name,
                    'raw_score': raw_score, 
                    'score_threshold': scoreThreshold}), 200

@app.route('/automated_attendance_testing/stop', methods=['POST'])
def automated_attendance_testing_stop():
    '''
    End of automated testing
    '''
    print("INFO: Stopping Automated Attendance Testing")
    return jsonify({'message': system.automatedAttendanceTestingStop()}), 200


if __name__ == '__main__':
    app.run(debug=True) 