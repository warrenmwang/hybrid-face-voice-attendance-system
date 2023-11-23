import numpy as np
import librosa
import matplotlib.pyplot as plt
import os
import cv2
import soundfile

SAMPLE_RATE=48000

class SpeakerRecognition:
    def readAudioFileIntoNpNdArray(self, filename:str)->np.ndarray:
        '''
        This function reads an audio file and converts it into a numpy ndarray.
        
        Inputs:
        filename (str): The name of the audio file to be read.
        
        Returns:
        np.ndarray: A numpy ndarray representation of the audio file.
        '''
        audio, _ = librosa.load(filename, sr=None)
        return audio

    def saveAudioNpNdArrayAsFile(self, audio:np.ndarray, filename:str):
        '''
        This function saves a numpy ndarray representation of an audio file.
        
        Inputs:
            audio (np.ndarray): The numpy ndarray representation of the audio file.
                if using wave module, note that sound should be as dtype int32
            filename (str): The name of the audio file to be saved.
        '''
        # FIXME: fuck it let's ball -- use double the sample_rate bc that is what is working experimentally, but idk why.
        soundfile.write(filename, audio, SAMPLE_RATE*2)

    def create_melspectrogram(self, audio_sample : np.ndarray) -> np.ndarray:
        '''
        audio_sample should have dtype float64, is a time series audio data
        if audio_sample is int32 dtype, will convert to float64
        '''
        if str(audio_sample.dtype) != "float64":
            audio_sample = audio_sample.astype(float)
        
        mel_bands = 128

        D = np.abs(librosa.stft(audio_sample))**2
        S = librosa.feature.melspectrogram(S=D, sr=SAMPLE_RATE, n_mels=mel_bands)
        S_dB = librosa.power_to_db(S, ref=np.max)
        return S_dB
    
    def convertAudioToImageRepresentation(self, audio : np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        '''
        create spectrogram and convert it into an RGB image

        Inputs:
            audio (np.ndarray): one dimensional (mono) array of ints

        return spectrogram and image
        - spectrogram is a numpy array
        - image is a numpy array
        '''
        tmp_filename = 'plot_without_axes.png'

        # create spectrogram (float64 values)
        mel_spectrogram = self.create_melspectrogram(audio)

        # convert into an image
        fig, ax = plt.subplots()
        ax.imshow(mel_spectrogram)
        ax.axis('off')
        fig.savefig(tmp_filename, bbox_inches='tight', pad_inches=0, format='png')
        plt.close(fig)

        # load image back as an numpy array (as color format BGR which is what we're using anyways)
        image = cv2.imread(tmp_filename)  
        
        # delete the file
        os.remove(tmp_filename)
        return mel_spectrogram, image
