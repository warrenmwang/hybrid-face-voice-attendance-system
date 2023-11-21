import numpy as np
import librosa
import matplotlib.pyplot as plt
import os
import cv2

class SpeakerRecognition:
    def create_melspectrogram(self, audio_sample : np.ndarray) -> np.ndarray:
        '''
        audio_sample should have dtype float64, is a time series audio data
        if audio_sample is int32 dtype, will convert to float64
        '''
        if str(audio_sample.dtype) != "float64":
            audio_sample = audio_sample.astype(float)
        
        sample_rate = 48000
        mel_bands = 128

        D = np.abs(librosa.stft(audio_sample))**2
        S = librosa.feature.melspectrogram(S=D, sr=sample_rate, n_mels=mel_bands)
        S_dB = librosa.power_to_db(S, ref=np.max)
        return S_dB
    
    def convertAudioToImageRepresentation(self, audio : np.ndarray) -> (np.ndarray, np.ndarray):
        '''
        create spectrogram and convert it into an RGB image

        return spectrogram and image
        - spectrogram is a numpy array
        - image is a numpy array
        '''
        tmp_filename = 'plot_without_axes.png'

        # create spectrogram (float64 values)
        mel_spectrogram = self.create_melspectrogram(audio)

        # convert into an RGB image
        fig, ax = plt.subplots()
        ax.imshow(mel_spectrogram)
        ax.axis('off')
        fig.savefig(tmp_filename, bbox_inches='tight', pad_inches=0, format='png')
        plt.close(fig)

        # load image back as an numpy array
        image = cv2.imread(tmp_filename)  
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # matplotlib uses rgb, cv2 uses bgr, convert from bgr to rgb
        
        # delete the file
        os.remove(tmp_filename)
        return mel_spectrogram, image
