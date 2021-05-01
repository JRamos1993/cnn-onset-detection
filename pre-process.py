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

def list_audio_files(data_dir):
    audio_dir = join(data_dir, 'audio')
    return [join(audio_dir, f) for f in sorted(listdir(audio_dir))]

def list_annotation_files(data_dir):
    ann_dir = join(data_dir, 'annotations', 'onsets')
    return [join(ann_dir, f) for f in sorted(listdir(ann_dir))]

def get_img_from_fig(fig, dpi=180):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi)
    buf.seek(0)
    img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    buf.close()
    img = cv2.imdecode(img_arr, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def read_img():
    # Read image
    img = Image.open('images\\onsets\\A0-F2.png').convert('RGB').resize((80,15))
    image = np.array(img)
    print(image.shape)
    # Output Images
    img.show()

def main():
    print('Starting script for pre-processing...')
    # onsets_images_dir = join('dataset_transformed', 'train')# , 'onsets')
    # non_onsets_images_dir = join('dataset_transformed', 'train')# , 'non-onsets')
    onsets_images_dir = 'dataset_transformed'
    non_onsets_images_dir = 'dataset_transformed'

    dataset_dir = 'dataset'
    audio_files = list_audio_files(dataset_dir)
    ann_files = list_annotation_files(dataset_dir)
    frame_size = 4096

    print('There are ' + str(len(audio_files)) + ' audio files and ' + str(len(ann_files)) + 'annotation files')

    i = 0
    onsets = 0
    non_onsets = 0
    for audio_file in audio_files:
        file_name = basename(audio_file)
        print('Pre-processing file ' + str(i) + '/' + str(len(audio_files)) + ': ' + file_name)

        # Read audio file
        sig = Signal(audio_file, sample_rate = 44100, num_channels = 1)

        # Read onset annotations for current audio file
        onset_file = ann_files[i]
        onsets = np.loadtxt(onset_file)
        print('Onsets read from ' + onset_file)

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
            else:
                # We continue from where we stopped last time
                f = len(frames) - 1

        # Each frame has nearly 93ms of audio
        # t = (frame_size / sample_rate) * 1000 = (4096 / 44100) * 1000
        t = 0.09287981859410431
        start = 0
        end = t
        f = 0
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

            if hasOnset:
                print('There is an onset within the range: ' + str(start) + ' to ' + str(end) + 'ms')
                onsets += 1
            else:
                print('There are no onsets within the range: ' + str(start) + ' to ' + str(end) + 'ms')
                non_onsets += 1

            # Apply CWT
            time = np.arange(frame_size, dtype=np.float16)
            scales = np.arange(1,81) # scaleogram with 80 rows
            cwt = scg.CWT(time, frame, scales)
            # print(cwt.coefs.shape)

            # Get scaleogram
            ax = scg.cws(cwt, yaxis='frequency', wavelet='morl', cbar=None)
            
            # Remove axis from image
            plt.subplots_adjust(bottom = 0, top = 1, left = 0, right = 1)
            # plt.show()

            # Get image from matplot and process it
            fig = plt.gcf()
            plot_img_np = get_img_from_fig(fig)
            image = Image.fromarray(plot_img_np).convert('RGB').resize((80,15))

            # Save image
            if hasOnset:
                image.save(onsets_images_dir + '\\' + '1-' + file_name + '-F' + str(f) + '.png')
            else:
                image.save(non_onsets_images_dir + '\\' + '0-' + file_name + '-F' + str(f) + '.png')

            plt.close()

        i += 1

    print('Total of ' + str(i) + ' frames ')
    print('From which ' + str(onsets) + ' contains onsets and ' + str(non_onsets) + ' do not contain any onset')

if __name__ == '__main__':
    main()
