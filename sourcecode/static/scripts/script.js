// ------------------------------------ Shared functions ------------------------------------
function getRadioSelectedValue(radios) {
    // gets the selected value out of a multiple choice radio button table
    for (var i = 0; i < radios.length; i++) {
        if (radios[i].checked) {
            return radios[i].value;
        }
    }
}

function removeAddedElementsFromList(list){
    // remove from whatever div all the elements in list
    let len = list.length;
    for (let i = 0; i < len; i++) {
        let elementToRemove = list.pop();
        elementToRemove.remove();
    }
}

// start continuous video stream for specified video display
function startContinuousVideoStream(divId, audioToggle = false) {
    var video = document.getElementById(divId);
    // Get access to the camera to prepare for continuous feed
    if(navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {

        if(audioToggle){
            // with audio
            video.muted = true; // to prevent the audio from playing back when the video is on, audio will still be recorded
            navigator.mediaDevices.getUserMedia({ video: true , audio: { sampleRate: 48000 }}).then(function(mediaStream) {
                stream = mediaStream;
                video.srcObject = stream;
                video.play();
            });
        }else{
            // without audio
            navigator.mediaDevices.getUserMedia({ video: true}).then(function(mediaStream) {
                stream = mediaStream;
                video.srcObject = stream;
                video.play();
            });
        }
    }
}

// stop continuous video stream
function stopContinuousVideoStream() {
    if (stream) {
        stream.getTracks().forEach(function(track) {
            track.stop();
        });
    }
}

// transitions between the different screens (main menu, enrollment, attendance, plot attendance)
function transitionScreens(hideId, showId) {
    var hideScreen = document.getElementById(hideId);
    var showScreen = document.getElementById(showId);

    hideScreen.classList.add('hidden');
    setTimeout(function() {
        hideScreen.style.display = 'none';
        showScreen.style.display = 'block';
        setTimeout(function() {
            showScreen.classList.remove('hidden');
        }, 50);
    }, 1000);
}

// acts like time.sleep(ms) in python
function delay(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
}
// ------------------------------------ Shared functions ------------------------------------

// MAIN MENU
var stream; // keep track of a continuous video stream for enroll/attendance, need reference to close when returning to main menu

// from main menu to another screen
document.getElementById('goToEnrollmentScreen').addEventListener('click', function() {
    transitionScreens('main_screen', 'enrollment_screen');
    startContinuousVideoStream('enrollmentVideo', true);
});
document.getElementById('goToRecordAttendanceScreen').addEventListener('click', function() {
    transitionScreens('main_screen', 'record_attendance_screen');
    startContinuousVideoStream('attendanceVideo');
});
document.getElementById('goToPlotAttendanceScreen').addEventListener('click', function() {
    transitionScreens('main_screen', 'plot_attendance_screen');
});
document.getElementById('goToAdminScreen').addEventListener('click', function() {
    transitionScreens('main_screen', 'admin_screen');
})

// from another screen to main menu        
document.getElementById('goToMainMenuFromEnrollment').addEventListener('click', function() {
    transitionScreens('enrollment_screen', 'main_screen');
    stopContinuousVideoStream();
});
document.getElementById('goToMainMenuFromRecordAttendance').addEventListener('click', function() {
    transitionScreens('record_attendance_screen', 'main_screen');
    stopContinuousVideoStream();
});
document.getElementById('goToMainMenuFromPlotAttendance').addEventListener('click', function() {
    transitionScreens('plot_attendance_screen', 'main_screen');
});
document.getElementById('goToMainMenuFromAdmin').addEventListener('click', function() {
    transitionScreens('admin_screen', 'main_screen');
})

// ---------------------------- ENROLLMENT ----------------------------
let enrollmentAddedImageElements = [];

document.getElementById('enrollmentClearDisplayedPictures').addEventListener('click', function() {
    removeAddedElementsFromList(enrollmentAddedImageElements);
});

document.getElementById('enrollmentTakePicture').addEventListener('click', function () {
    // takes the username and takes the picture, sends both to the backend.

    var inputField = document.getElementById('enrollmentUserNameInput');
    var username = inputField.value;
    if (username === '') {
        alert('Name cannot be empty.');
        return;
    }

    // get a frame from the video feed
    var video = document.getElementById('enrollmentVideo');
    var canvas = document.createElement('canvas');
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    var ctx = canvas.getContext('2d');
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    var imageDataURL = canvas.toDataURL('image/png');

    // Create an image element to display the captured picture
    var capturedImage = document.createElement('img');
    capturedImage.src = imageDataURL;
    var displayDiv = document.getElementById('enrollment_img1_display');
    var br = document.createElement("br");
    displayDiv.appendChild(capturedImage);
    displayDiv.appendChild(br);
    enrollmentAddedImageElements.push(capturedImage);
    enrollmentAddedImageElements.push(br);

    // to be received by python backend to do arbitrary manipulation (greyscale for now)
    // Convert data URL to Blob
    fetch(imageDataURL)
        .then(res => res.blob())
        .then(blob => {
            // Create a FormData object
            var formData = new FormData();
            formData.append('image', blob);

            // add username
            formData.append('text', username);

            // Send the image and username to the server
            fetch('/enrollment', {
                method: 'POST',
                body: formData
            })
            .then(response => response.blob())
            .then(imageBlob => {
                // Create a new image element
                var manipulatedImage = document.createElement('img');

                // Create a URL for the image Blob
                var imageUrl = URL.createObjectURL(imageBlob);

                // Set the image source to the URL
                manipulatedImage.src = imageUrl;

                // Add the image to the page
                var displayDiv = document.getElementById('enrollment_img2_display');
                var br = document.createElement("br");
                displayDiv.appendChild(manipulatedImage);
                displayDiv.appendChild(br);
                enrollmentAddedImageElements.push(manipulatedImage);
                enrollmentAddedImageElements.push(br);
            });
        });
});

document.getElementById('enrollmentImageUploadButton').addEventListener('click', function () {
    var inputField = document.getElementById('enrollmentUserNameInput');
    var username = inputField.value;

    const fileInput = document.getElementById('enrollmentFileUpload');
    const file = fileInput.files[0];
    const formData = new FormData();

    if (!file) {
        alert('Please select an image');
        return;
    }

    if (!file.type.startsWith('image/')) { // Ensure it's an image
        alert('Only images are allowed');
        return;
    }

    // display the uploaded image
    var capturedImage = document.createElement('img');
    capturedImage.src = URL.createObjectURL(file);
    var displayDiv = document.getElementById('enrollment_img1_display');
    var br = document.createElement("br");
    displayDiv.appendChild(capturedImage);
    displayDiv.appendChild(br);
    enrollmentAddedImageElements.push(capturedImage);
    enrollmentAddedImageElements.push(br);
  
    // add data to be sent to backend for enrollment
    formData.append('image', file);
    formData.append('text', username);

    fetch('/enrollment', {
        method: 'POST',
        body: formData
    })
    .then(response => response.blob())
    .then(imageBlob => {
        // display the SIFT processed image
        // Create a new image element
        var manipulatedImage = document.createElement('img');

        // Create a URL for the image Blob
        var imageUrl = URL.createObjectURL(imageBlob);

        // Set the image source to the URL
        manipulatedImage.src = imageUrl;

        // Add the image to the page
        var displayDiv = document.getElementById('enrollment_img2_display');
        var br = document.createElement("br");
        displayDiv.appendChild(manipulatedImage);
        displayDiv.appendChild(br);
        enrollmentAddedImageElements.push(manipulatedImage);
        enrollmentAddedImageElements.push(br);
    });
});

let mediaRecorder;
let enrollmentRecordedChunks = [];
let enrollmentHybridVideoDisplayElements = [];

document.getElementById('enrollmentHybridClearDisplay').addEventListener('click', function() {
    removeAddedElementsFromList(enrollmentHybridVideoDisplayElements);
})

// enrollment hybrid event listeners
document.getElementById('enrollmentStartHybridRecording').addEventListener('click', enrollmentHybridStartRecording);
document.getElementById('enrollmentStopHybridRecording').addEventListener('click', enrollmentHybridStopRecording);

// try sending both a video stream and an audio stream

// start recording
function enrollmentHybridStartRecording() {
    if (!stream) {
        console.error('No video stream up and running');
        return;
    }

    var inputField = document.getElementById('enrollmentUserNameInput');
    var username = inputField.value;
    if (username === '') {
        alert('Name cannot be empty.');
        return;
    }

    mediaRecorder = new MediaRecorder(stream);

    mediaRecorder.addEventListener('dataavailable', event => {
        enrollmentRecordedChunks.push(event.data);
    })

    mediaRecorder.start();
}

// stop recording and then send
function enrollmentHybridStopRecording() {
    var inputField = document.getElementById('enrollmentUserNameInput');
    var username = inputField.value;
    if (username === '') {
        alert('Name cannot be empty.');
        return;
    }

    mediaRecorder.addEventListener('stop', () => {
        let enrollmentHybridDisplay = document.getElementById('enrollmentHybridDisplay');
        var br = document.createElement('br');
        
        // convert the recorded video+audio chunks into single blob
        const recordedBlob = new Blob(enrollmentRecordedChunks, { type: 'video/webm'});
        enrollmentRecordedChunks = []; // clear data buffer

        // create video element to be displayed directly in html
        const videoElement = document.createElement('video');

        // Create a URL from the blob
        const url = URL.createObjectURL(recordedBlob);

        // Set the source of the video element to the URL
        videoElement.src = url;

        // Set the controls attribute so the user can control the video
        videoElement.controls = true;

        // display the captured video+audio
        enrollmentHybridDisplay.appendChild(videoElement);
        enrollmentHybridDisplay.appendChild(br);
        enrollmentHybridVideoDisplayElements.push(videoElement);
        enrollmentHybridVideoDisplayElements.push(br);

        // Create a FormData object
        var formData = new FormData();

        // add video and audio
        formData.append('videoAndAudio', recordedBlob);

        // add username
        formData.append('text', username);

        // send to backend
        fetch('/enrollment/hybrid', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(imagesBase64 => {
            // order of images received is [random orig frame, extracted feature frame, mel spectrogram of voice]
            let imgs = [];
            imagesBase64.forEach(base64 => {
                let img = document.createElement('img');
                img.src = 'data:image/jpeg;base64,' + base64;
                imgs.push(img);
            })

            // add them to the screen
            document.getElementById('enrollmentHybrid_img1_display').appendChild(imgs[0]);
            document.getElementById('enrollmentHybrid_img1_display').appendChild(br);
            enrollmentHybridVideoDisplayElements.push(imgs[0]);
            enrollmentHybridVideoDisplayElements.push(br);

            document.getElementById('enrollmentHybrid_img2_display').appendChild(imgs[1]);
            document.getElementById('enrollmentHybrid_img2_display').appendChild(br);
            enrollmentHybridVideoDisplayElements.push(imgs[1]);
            enrollmentHybridVideoDisplayElements.push(br);

            document.getElementById('enrollmentHybrid_img3_display').appendChild(imgs[2]);
            document.getElementById('enrollmentHybrid_img3_display').appendChild(br);
            enrollmentHybridVideoDisplayElements.push(imgs[2]);
            enrollmentHybridVideoDisplayElements.push(br);
        });
    });

    mediaRecorder.stop();
}


// RECORD ATTENDANCE
let featureExtractors = ['SIFT', 'CNN', 'VGG16'];
let takingAttendanceFlag = false;
let attendanceMarkedPresentElements = [];
let attendanceAutomatedTestingElements = [];
let attendanceAutomatedTestingFinalResultElements = [];

document.getElementById('clearAttendanceMarkedPresentField').addEventListener('click', function () {
    removeAddedElementsFromList(attendanceMarkedPresentElements);
});

function addThreeColumnRowToTable(table, trackList, col1, col2, col3) {
    // adds a row with the contents col1, col2, col3 into the given table
    // also adds the new row into the trackList to keep a reference to delete the row later
    var row = document.createElement('tr');
    var cell1 = document.createElement('td');
    var cell2 = document.createElement('td');
    var cell3 = document.createElement('td');
    cell1.innerHTML = col1;
    cell2.innerHTML = col2;
    cell3.innerHTML = col3;
    row.appendChild(cell1);
    row.appendChild(cell2);
    row.appendChild(cell3);
    table.appendChild(row);
    trackList.push(row);
}

var intervalId;
document.getElementById('startTakingAttendance').addEventListener('click', function() {
    // check flag ensure another "worker" cannot be started
    if (takingAttendanceFlag == true){
        return;
    }
    takingAttendanceFlag = true;

    // toggle indicator "light" to show taking attendance
    document.getElementById('takingAttendanceIndicator').classList.remove('off');
    document.getElementById('takingAttendanceIndicator').classList.add('on');

    // Set the interval for sending frames to the backend
    var frameInterval = 1000; // 1000ms = 1s
    var video = document.getElementById('attendanceVideo');
    var radiosField = document.getElementsByName('feature_extraction');
    var scoreThresholdField = document.getElementById('attendanceScoreThresholdInput');
    var featureExtractionMethod;
    var scoreThreshold;

    intervalId = setInterval(function() {
        // get a frame from the video feed
        var canvas = document.createElement('canvas');
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        var ctx = canvas.getContext('2d');
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
        var imageDataURL = canvas.toDataURL('image/png');

        // take attendance based off of selected feature extraction method (sift, cnn, pretrained model)
        featureExtractionMethod = getRadioSelectedValue(radiosField);
        // similarity score threshold
        scoreThreshold = scoreThresholdField.value;

        // send frame to backend
        fetch(imageDataURL)
            .then(res => res.blob())
            .then(blob => {
                var formData = new FormData();
                formData.append('image', blob);
                formData.append('featureExtractionMethod', featureExtractionMethod);
                formData.append('scoreThreshold', scoreThreshold)

                fetch('/attendance',  {
                    method: 'POST',
                    body: formData
                })
                .then(response => response.json())
                .then(data => {
                    if (data.name) {
                        // show who was marked present if message is not Null
                        // expect (name, raw_score, score_threshold)
                        var table = document.getElementById('attendanceMarkedPresentField');
                        addThreeColumnRowToTable(table, attendanceMarkedPresentElements, data.name, data.raw_score, data.score_threshold);
                    }
                });
            });

    }, frameInterval);
});

document.getElementById('stopTakingAttendance').addEventListener('click', function() {
    // check flag ensure another "worker" cannot be started
    if (takingAttendanceFlag == false){
        return;
    }
    takingAttendanceFlag = false;

    // toggle indicator "light" to show taking attendance
    document.getElementById('takingAttendanceIndicator').classList.remove('on');
    document.getElementById('takingAttendanceIndicator').classList.add('off');

    // stop sending frames to backend
    clearInterval(intervalId);
});

document.getElementById('resetTakingAttendance').addEventListener('click', function() {
    fetch('/reset_attendance', {
        method: 'POST'
    });
})

document.getElementById('automatedAttendanceTestingStart').addEventListener('click', async function() {
    var automatedAttendanceTestingDisplayField = document.getElementById('automatedAttendanceTestingDisplayField');
    var processResultsTable = document.getElementById('automatedAttendanceTestingDisplayField');
    var br = document.createElement('br');
    var video = document.getElementById('attendanceVideo');
    var delayTime = 100; // in milliseconds delay for each iteration of taking pictures and processing

    // get number of iterations per method to run and the ground truth name of person being run against
    var numItersInputField = document.getElementById('automatedAttendanceTestingNumIters');
    var groundTruthInputField = document.getElementById('automatedAttendanceTestingGroundTruth');
    var scoreThresholdInputField_SIFT = document.getElementById('automatedAttendanceTestingScoreThresholdInput_SIFT');
    var scoreThresholdInputField_CNN = document.getElementById('automatedAttendanceTestingScoreThresholdInput_CNN');
    var scoreThresholdInputField_VGG16 = document.getElementById('automatedAttendanceTestingScoreThresholdInput_VGG16');
    
    var numItersPerMethod = numItersInputField.value;
    var groundTruth = groundTruthInputField.value;
    var scoreThreshold_SIFT = scoreThresholdInputField_SIFT.value;
    var scoreThreshold_CNN = scoreThresholdInputField_CNN.value;
    var scoreThreshold_VGG16 = scoreThresholdInputField_VGG16.value;

    // ping backend to start recording of automated test
    var formData = new FormData();
    formData.append('scoreThreshold_SIFT', scoreThreshold_SIFT);
    formData.append('scoreThreshold_CNN', scoreThreshold_CNN);
    formData.append('scoreThreshold_VGG16', scoreThreshold_VGG16);
    formData.append('groundTruth', groundTruth);
    formData.append('numItersPerMethod', numItersPerMethod);
    fetch('/automated_attendance_testing/start', {
        method: 'POST',
        body: formData
    });

    // loop thru each feature extraction method N times
    for (let featureExtractionMethod of featureExtractors){
        addThreeColumnRowToTable(processResultsTable, attendanceAutomatedTestingElements, featureExtractionMethod, '--', '--');
        for (let i = 0; i < numItersPerMethod; i++){
            // get frame from video feed
            var canvas = document.createElement('canvas');
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
            var ctx = canvas.getContext('2d');
            ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
            var imageDataURL = canvas.toDataURL('image/png');

            // send image, featureExtractionMethod, and groundTruth to be processed by backend
            fetch(imageDataURL)
                .then(res => res.blob())
                .then(blob => {
                    var formData = new FormData();
                    formData.append('image', blob);
                    formData.append('featureExtractionMethod', featureExtractionMethod);
                    fetch('/automated_attendance_testing/process', {
                        method: 'POST',
                        body: formData
                    }).then(response => response.json())
                    .then(data => {
                        // each response is the individual feature extraction method response
                        addThreeColumnRowToTable(processResultsTable, attendanceAutomatedTestingElements, data.name, data.raw_score, data.score_threshold);
                });
            });

            // delay
            await delay(delayTime * 4);
            }
    }

    // ping backend to stop recording automated test and get final results
    fetch('/automated_attendance_testing/stop', {
        method: 'POST'
    }).then(response => response.json())
    .then(data => {
        // show the final results
        var finalResultDisplay = document.getElementById('automatedAttendanceTestingFinalResultsDisplayField');
        var finalResultsText = document.createElement('div');
        finalResultsText.innerHTML = data.message;
        finalResultDisplay.appendChild(finalResultsText);
        attendanceAutomatedTestingFinalResultElements.push(finalResultsText);
    });
});

document.getElementById('automatedAttendanceTestingClearFinalResultsDisplay').addEventListener('click', function() {
    removeAddedElementsFromList(attendanceAutomatedTestingFinalResultElements);
});

document.getElementById('automatedAttendanceTestingClearDisplay').addEventListener('click', function() {
    removeAddedElementsFromList(attendanceAutomatedTestingElements);
});

// PLOT ATTENDANCE
let plotAttendanceAddedImageElements = [];

document.getElementById('createAttendanceHistogram').addEventListener('click', function() {
    fetch('/plot_attendance_histogram', {
        method: 'GET'
    })
    .then(response => response.blob())
    .then(imageBlob => {
        var histogramImage = document.createElement('img');
        var imageUrl = URL.createObjectURL(imageBlob);
        histogramImage.src = imageUrl;
        var displayDiv = document.getElementById('plotAttendanceImageDisplay');
        var br = document.createElement('br');
        displayDiv.appendChild(histogramImage);
        displayDiv.appendChild(br);
        plotAttendanceAddedImageElements.push(histogramImage);
        plotAttendanceAddedImageElements.push(br);
    })
});

document.getElementById('clearAttendanceImages').addEventListener('click', function() {
    removeAddedElementsFromList(plotAttendanceAddedImageElements);
});

document.getElementById('clearAttendanceCSVFile').addEventListener('click', function() {
    fetch('/clear_attendance_file', {
        method: 'POST'
    });
});

// ADMIN page
let adminPageAddedElements = [];
document.getElementById('reEnrollAllCurrentUsers').addEventListener('click', function() {
    var adminDisplayField = document.getElementById('adminDisplayField');
    var br = document.createElement('br');

    // tell user to wait
    var textNode = document.createTextNode('Waiting on confirmation that process is complete...');
    adminDisplayField.appendChild(textNode);
    adminDisplayField.appendChild(br);
    adminPageAddedElements.push(textNode);
    adminPageAddedElements.push(br);

    // tell backend to recompute features for all images
    fetch('/re-extract-all-images', {
        method: 'POST'
    })
    .then(response => response.json())
    .then(data => {
        var textNode = document.createTextNode(data.message);
        adminDisplayField.appendChild(textNode);
        adminDisplayField.appendChild(br);
        adminPageAddedElements.push(textNode);
        adminPageAddedElements.push(br);
    });
});

document.getElementById('adminClear').addEventListener('click', function() {
    removeAddedElementsFromList(adminPageAddedElements);
})
