import glob
import os
import numpy as np
import scipy.misc
import librosa
from PIL import Image

name = ['01_speech', '02_song']

Width = 650
Height = 180

count = 0.
garos = 0.
passcounter = 0


for fname in name :
    if not os.path.exists('RAVDESS/Mels') :
        os.mkdir('RAVDESS/Mels')

    if not os.path.exists('RAVDESS/Mels/%s'%(fname)) :
        os.mkdir('RAVDESS/Mels/%s'%(fname))

    folders = np.array([])
    index = 0
    for name in glob.glob('RAVDESS/%s/*'%(fname)) :
        folders = np.append(folders,name)
        a,b = folders[index].split('RAVDESS/%s'%(fname))
        if not os.path.exists(('RAVDESS/Mels/%s'%(fname)+b)) :
            os.mkdir(('RAVDESS/Mels/%s'%(fname)+b))
        index += 1

    index = 1

    for label in folders :
        for files in glob.glob(str(label + '/*.wav')) :
            a,b = label.split('RAVDESS/%s/'%(fname))

            audio_path_num = files
            # audio_path = '/path/to/your/favorite/song.mp3'

            c, d = files.split(label)
            audioname,e = d.split('.wav')

            y, sr = librosa.load(audio_path_num, sr=48000)

            # cutter = len(y) / sr
            #
            # plz = cutter * sr
            # whatthe = y.shape
            #
            # if y.size > cutter * sr:
            #     y = y[0:sr]

            # Normalize
            librosa.util.normalize(y, norm=1)

            # Let's make a spectrogram (freq, power)
            Spec = librosa.amplitude_to_db(abs(librosa.stft(y, n_fft=2048)), ref=np.max)

            # Let's make a CQT
            C = librosa.amplitude_to_db(abs(librosa.cqt(y, sr=sr)))

            # Let's make and display a mel-scaled power (energy-squared) spectrogram
            S = librosa.feature.melspectrogram(y, sr=sr, n_mels=128)

            # Convert to log scale (dB). We'll use the peak power (max) as reference.
            log_S = librosa.power_to_db(S, ref=np.max)

            # Next, we'll extract the top 13 Mel-frequency cepstral coefficients (MFCCs)
            mfcc = librosa.feature.mfcc(S=log_S, n_mfcc=13)

            # Let's pad on the first and second deltas while we're at it
            delta_mfcc = librosa.feature.delta(mfcc)
            delta2_mfcc = librosa.feature.delta(mfcc, order=2)

            if len(Spec[0]) == len(C[0]) == len(S[0]) == len(log_S[0]) == len(mfcc[0]) :
                rawdata_length = len(Spec[0])
            else :
                print('Input Size Different..?')
                quit()

            rawdata_height = y.size/rawdata_length

            # Raw Data
            rawdata_array = y[0:(rawdata_height) * rawdata_length]
            rawdata_array = np.reshape(rawdata_array, (rawdata_height, rawdata_length))

            scipy.misc.imsave('1.raw.png', np.flipud(rawdata_array))
            scipy.misc.imsave('2.spec.png', np.flipud(Spec))
            scipy.misc.imsave('3.cqt.png', np.flipud(C))
            scipy.misc.imsave('4.mel_power_spectrogram.png', np.flipud(log_S))
            scipy.misc.imsave('5.mf.png', np.flipud(mfcc))
            scipy.misc.imsave('6.mf_d.png', np.flipud(delta_mfcc))
            scipy.misc.imsave('7.mf_dd.png', np.flipud(delta2_mfcc))

            # Let's Concat
            # raw_img = Image.open('1.raw.png')
            # spec_img = Image.open('2.spec.png')
            # cqt_img = Image.open('3.cqt.png')
            mel_spec_img = Image.open('4.mel_power_spectrogram.png')
            mel_img = Image.open('5.mf.png')
            mel_d_img = Image.open('6.mf_d.png')
            mel_dd_img = Image.open('7.mf_dd.png')

            Input_stack = np.vstack((mel_spec_img, mel_img, mel_d_img, mel_d_img, mel_dd_img))
            # zero padd image, PADDING SIZE : 600, 1800
            Input_PAD = np.zeros((Height, Width), dtype=np.float64)
            h = Input_stack.shape[0]
            w = Input_stack.shape[1]

            if w > Width :
                print ('file : %s passed -> legnth : %d'%(audioname,w))
                passcounter += 1
                continue

            x_put = (Width - w) / 2
            y_put = (Height - h) / 2

            if h % 2 == 0 and y_put != 0:
                y_put - 1
            if w % 2 == 0 and x_put != 0:
                x_put - 1

            Input_PAD[y_put:y_put + h][:, x_put:x_put + w] = Input_stack

            Input = Input_PAD

            garos += w
            count += 1


            # str('{:03}'.format(index))
            filename = 'RAVDESS/Mels/%s/'%(fname)+b+ '/' + audioname + '.png'
            scipy.misc.imsave(filename, Input)
            index += 1
            print ('%s is saved' % (filename))


mean = 0.
mean = garos/count

print('length mean : %f'%(mean))
print('Passnum : %d'%(passcounter))



