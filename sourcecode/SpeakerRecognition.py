import numpy as np
import librosa

def create_melspectrogram(audio_sample : np.ndarray) -> np.ndarray:
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