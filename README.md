## Hybrid Face and Voice Recognition Attendance System
### Author: Warren Wang

Web ui for a hybrid face and voice recognition attendance system. 
Originally written for a college elective class on Introduction to Biometrics. 

### Getting Started
0. Prerequisites
- Requires `ffmpeg` installed for the audio processing
    - If on ubuntu: `sudo apt install ffmpeg`
1. Clone Repo
```
git clone git@github.com:warrenmwang/hybrid-face-voice-attendance-system.git
cd hybrid-face-voice-attendance-system
```
2. Setup environment
```
conda create -n ENV_NAME python=3.11.0
conda activate ENV_NAME
pip install -r requirements.txt
```
3. Create .env with a directory for the database
```
echo "DB=./database" > .env
```
4. Run
```
python ./Main.py
```

Go to localhost:5000, uses Flask development web server as default.

### Tech stack

Front-end: HTML, CSS, JS

Back-end: Python (Flask, OpenCV, Tensorflow/Keras, matplotlib, numpy, etc.)
