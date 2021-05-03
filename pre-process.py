import numpy as np
from scipy import signal
import scaleogram as scg
from os.path import basename, exists, join, splitext
from os import listdir
from scipy.io.wavfile import read
import soundfile as sf
import matplotlib.pyplot as plt
from madmom.audio.signal import FramedSignal, Signal
from PIL import Image
import matplotlib.image as mpimg
import io
import cv2
import glob
import pywt
import librosa
import librosa.display
from madmom.audio.filters import MelFilterbank
from madmom.audio.spectrogram import (FilteredSpectrogram, Spectrogram,
                                      LogarithmicSpectrogram)
from madmom.audio.stft import ShortTimeFourierTransform

def list_audio_files(data_dir):
    audio_dir = join(data_dir, 'audio')
    return [join(audio_dir, f) for f in sorted(listdir(audio_dir))]

def list_annotation_files(data_dir):
    ann_dir = join(data_dir, 'annotations', 'onsets')
    return [join(ann_dir, f) for f in sorted(listdir(ann_dir))]

def get_img_from_fig(fig, dpi = 180):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi)
    buf.seek(0)
    img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    buf.close()
    img = cv2.imdecode(img_arr, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def get_list_of_continuous_wavelets():
    l = []
    for name in pywt.wavelist(kind='continuous'):
        # supress warnings when the wavelet name is missing parameters
        completion = {
            'cmor': 'cmor1.5-1.0',
            'fbsp': 'fbsp1-1.5-1.0',
            'shan': 'shan1.5-1.0' }
        if name in completion:
            name =  completion[name]# supress warning
        l.append( name+" :\t"+pywt.ContinuousWavelet(name).family_name )
    return l

def main():
    print('Starting script for pre-processing...')
    # onsets_images_dir = join('dataset_transformed', 'train')# , 'onsets')
    # non_onsets_images_dir = join('dataset_transformed', 'train')# , 'non-onsets')
    onsets_images_dir = 'dataset_transformed'
    non_onsets_images_dir = 'dataset_transformed'

    dataset_dir = 'dataset'
    audio_files = list_audio_files(dataset_dir)
    ann_files = list_annotation_files(dataset_dir)
    frame_size = 1024
    sample_rate = 44100
    t = frame_size / sample_rate
    # t = 0.09287981859410431 # seconds for frame_size = 4096

    time = np.arange(frame_size, dtype=np.float16)
    scales = np.arange(1,81) # scaleogram with 80 rows

    print('There are ' + str(len(audio_files)) + ' audio files and ' + str(len(ann_files)) + 'annotation files')

    i = 0
    for audio_file in audio_files:
        file_name = basename(audio_file)
        print('Pre-processing file ' + str(i) + '/' + str(len(audio_files)) + ': ' + file_name)

        # Read audio file
        sig = Signal(audio_file, sample_rate, num_channels = 1)

        # Read onset annotations for current audio file
        onset_file = ann_files[i]
        onsets = np.loadtxt(onset_file)
        print('Onsets read from ' + onset_file)
        number_of_onsets = len(onsets)
        print('There are ' + str(number_of_onsets) + ' onsets')

        # Split audio signal into frames of same size
        frames = FramedSignal(sig, frame_size, hop_size = frame_size)
        print('There are ' + str(len(frames)) + ' frames')

        # Check if we already generated the correct amount of frames for that file before
        matching_files = glob.glob('dataset_transformed/' + '*'+ file_name + '*')
        if len(matching_files) > 0:
            if len(frames) == len(matching_files):
                print('Skipping file ' + str(i) + '/' + str(len(audio_files)) + ': ' + file_name)
                i += 1
                continue

        start = 0
        end = t
        f = 0
        onsets_found_this_file = 0
        for frame in frames:
            # Plot frame
            # plt.plot(frame)
            # plt.show()

            # Check if contains onset
            start = f * t
            end = start + t
            f += 1
            hasOnset = False
            for onset in onsets:
                if start <= onset and end >= onset:
                    hasOnset = True
                    onsets_found_this_file += 1

            if hasOnset:
                print('There is an onset within the range: ' + str(start) + ' to ' + str(end) + 'ms')
            else:
                print('There are no onsets within the range: ' + str(start) + ' to ' + str(end) + 'ms')

            # Apply CWT
            cwt = scg.CWT(time, frame, scales, wavelet='cmor1.5-1.0')
            # print(cwt.coefs.shape)

            # Get scaleogram
            ax = scg.cws(cwt, yaxis = 'frequency', wavelet = 'cmor1.5-1.0', cbar = None, coi = False)

            # ['cgau1 :\tComplex Gaussian wavelets', 'cgau2 :\tComplex Gaussian wavelets', 
            # 'cgau3 :\tComplex Gaussian wavelets', 'cgau4 :\tComplex Gaussian wavelets', 
            # 'cgau5 :\tComplex Gaussian wavelets', 'cgau6 :\tComplex Gaussian wavelets', 
            # 'cgau7 :\tComplex Gaussian wavelets', 'cgau8 :\tComplex Gaussian wavelets', 
            # 'cmor1.5-1.0 :\tComplex Morlet wavelets', 'fbsp1-1.5-1.0 :\tFrequency B-Spline wavelets',
            #  'gaus1 :\tGaussian', 'gaus2 :\tGaussian', 'gaus3 :\tGaussian', 'gaus4 :\tGaussian', 
            #  'gaus5 :\tGaussian', 'gaus6 :\tGaussian', 'gaus7 :\tGaussian', 'gaus8 :\tGaussian', 
            #  'mexh :\tMexican hat wavelet', 'morl :\tMorlet wavelet', 'shan1.5-1.0 :\tShannon wavelets']

            # Remove axis from image
            plt.subplots_adjust(bottom = 0, top = 1, left = 0, right = 1)
            plt.show()

            # Get image from matplot and process it
            fig = plt.gcf()
            plot_img_np = get_img_from_fig(fig)
            image = Image.fromarray(plot_img_np).convert('RGB').resize((15,80))

            # Save image
            if hasOnset:
                image.save(onsets_images_dir + '\\' + '1-' + file_name + '-F' + str(f) + '.png')
            else:
                image.save(non_onsets_images_dir + '\\' + '0-' + file_name + '-F' + str(f) + '.png')

            plt.close()

        if number_of_onsets != onsets_found_this_file:
            print('It was supposed to have ' + str(number_of_onsets) + ' onsets. Found '+ str(onsets_found_this_file) + 'instead. Exiting...')
            exit()

        i += 1

if __name__ == '__main__':
    main()
