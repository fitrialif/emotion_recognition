import numpy as np
import os
from six.moves import xrange
from util import Util as util
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sn
import time
import copy

def softmax(z):
    assert len(z.shape) == 2
    s = np.max(z, axis=1)
    s = s[:, np.newaxis] # necessary step to do broadcasting
    e_x = np.exp(z - s)
    div = np.sum(e_x, axis=1)
    div = div[:, np.newaxis] # dito
    return e_x / div

def conf_matrix(score, label):
    score_ = np.expand_dims(np.argmax(score, axis=1), axis=1)
    conf = confusion_matrix(label, score_)
    return conf

ROOT = '/home/lemin/nas/DMSL/AFEW/weights/Feature_wise_Fusion_test'
SAVE = '/home/lemin/nas/DMSL/AFEW/Ensemble/180620/'
save_dir = 'per_label_ensemble_trial3'
print("File name : " + save_dir)
if not os.path.exists(SAVE + save_dir):
    os.mkdir(SAVE + save_dir)
# else:
#     raise FileExistsError

filename = 'test_scores.npz'
label_path = '/home/lemin/nas/DMSL/AFEW/NpzData/Test2017/y_DenseNet121_titu_fc2_fc1_no_overlap.npz'
label = util.load_from_npz(label_path)
index = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

'''
    Angry : 107, Disgust : 29, Fear : 62, Happy : 116, Neutral : 225, Sad : 68, Surprise : 46
    Agnry : 0 ~ 106, Disgust : 107 ~ 135, Fear : 136 ~ 197, Happy : 198 ~ 313, Neutral : 314 ~ 538, Sad : 539 ~ 606, Surprise : 607 ~ 652
'''

data_path = {
    'Angry': os.path.join(ROOT, '../../Audio/audio6/17_20h_30m_resnet50_rhf_r90__Spec_cut_040.npz'),
    # 'Angry': os.path.join(ROOT, 'Fusion_Test_ver2', filename),
    'Disgust': os.path.join(ROOT, 'CtxLSTM_Test_ver8', filename),
    # 'Fear': os.path.join(ROOT, 'CtxLSTM_Test_ver8', filename),
    'Fear': os.path.join(ROOT, '../../Audio/audio2/15_20h_35m_resnet50_rhf_r45__Mels_060.npz'), # fear1
    'Happy': os.path.join(ROOT, 'CtxTAGM_ver3', filename),
    'Neutral': os.path.join(ROOT, 'C3D_lstm_Test_ver1', filename),
    'Sad': os.path.join(ROOT, 'CtxLSTM_Test_ver3', filename),
    'Surprise': os.path.join(ROOT, 'CtxTAGM_ver3', filename),
    'overall': os.path.join(ROOT, 'Ensemble_max', filename)
}

label_scores = {
    'Angry': softmax(util.load_from_npz(data_path['Angry'])), # 50.38
    'Disgust': softmax(util.load_from_npz(data_path['Disgust'])), # 48.39
    'Fear': softmax(util.load_from_npz(data_path['Fear'])), # 48.39
    'Happy': softmax(util.load_from_npz(data_path['Happy'])), # 50.84
    'Neutral': softmax(util.load_from_npz(data_path['Neutral'])), # 47.01
    'Sad': softmax(util.load_from_npz(data_path['Sad'])), # 48.85
    'Surprise': softmax(util.load_from_npz(data_path['Surprise'])), # 50.84
    'overall': softmax(util.load_from_npz(data_path['overall'])) # 53.90
}

copy_scores = copy.deepcopy(label_scores)

def label2value(label_):
    if label_ == 'Angry':
        x = 0
    elif label_ == 'Disgust':
        x = 1
    elif label_ == 'Fear':
        x = 2
    elif label_ == 'Happy':
        x = 3
    elif label_ == 'Neutral':
        x = 4
    elif label_ == 'Sad':
        x = 5
    elif label_ == 'Surprise':
        x = 6
    return x

scores = []
label_weight = [10, 5, 5, 10, 1, 10, 10] # max
# label_weight = [10, 10, 10, 10, 10, 10, 10]
suppress = 0.1
print('Label weight : {}, Suprress : {}'.format(label_weight, suppress))

iter = 10

f = open(SAVE + save_dir + '/config.txt', 'a')
f.write('Label Weight : ' + str(label_weight) +  '\n')
f.write('per Iteration : {}\n'.format(iter))
for key, item in data_path.items():
    f.write('{} : {}\n'.format(key, item))
f.close()

# Scores
# for key, item in sorted(label_scores.items()):
#     print(key)
#     val = label2value(key)
#     label_scores[key][:, val] = label_weight[val] * label_scores[key][:, val]
#     scores.append(label_scores[key])

# # Softmax
# for key, item in sorted(label_scores.items()):
#     print(key)
#     val = label2value(key)
#     softscores = softmax(label_scores[key])
#     softscores[:, val] = label_weight[val] * softscores[:, val]
#     scores.append(softscores)

# for key, item in sorted(label_scores.items()):
#     print(key)
#     if key == 'Angry':
#         label_scores[key][:, 0] = label_weight[0] * label_scores[key][:, 0]
#         label_scores[key][:, 1] = suppress * label_scores[key][:, 1]
#         label_scores[key][:, 2] = suppress * label_scores[key][:, 2]
#         label_scores[key][:, 3] = suppress * label_scores[key][:, 3]
#         label_scores[key][:, 4] = 0.01 * label_scores[key][:, 4]
#         label_scores[key][:, 5] = suppress * label_scores[key][:, 5]
#         label_scores[key][:, 6] = suppress * label_scores[key][:, 6]
#     elif key == 'Disgust':
#         label_scores[key][:, 1] = label_weight[1] * label_scores[key][:, 1]
#         label_scores[key][:, 0] = 0.01 * label_scores[key][:, 0]
#         label_scores[key][:, 2] = 0.01 * label_scores[key][:, 2]
#         label_scores[key][:, 3] = 0.01 * label_scores[key][:, 3]
#         label_scores[key][:, 4] = 0.01 * label_scores[key][:, 4]
#         label_scores[key][:, 5] = 0.01 * label_scores[key][:, 5]
#         label_scores[key][:, 6] = 0.01 * label_scores[key][:, 6]
#     elif key == 'Fear':
#         label_scores[key][:, 2] = label_weight[2] * label_scores[key][:, 2]
#         label_scores[key][:, 1] = 2 * label_scores[key][:, 1]
#         label_scores[key][:, 0] = 0.01 * label_scores[key][:, 0]
#         label_scores[key][:, 3] = 0.01 * label_scores[key][:, 3]
#         label_scores[key][:, 4] = 0.01 * label_scores[key][:, 4]
#         label_scores[key][:, 5] = 0.01 * label_scores[key][:, 5]
#         label_scores[key][:, 6] = 0.01 * label_scores[key][:, 6]
#     elif key == 'Happy':
#         label_scores[key][:, 3] = label_weight[3] * label_scores[key][:, 3]
#         label_scores[key][:, 1] = suppress * label_scores[key][:, 1]
#         label_scores[key][:, 2] = suppress * label_scores[key][:, 2]
#         label_scores[key][:, 0] = suppress * label_scores[key][:, 0]
#         label_scores[key][:, 4] = 0.1 * label_scores[key][:, 4]
#         label_scores[key][:, 5] = suppress * label_scores[key][:, 5]
#         label_scores[key][:, 6] = suppress * label_scores[key][:, 6]
#     elif key == 'Neutral':
#         label_scores[key][:, 4] = label_weight[4] * label_scores[key][:, 4]
#         label_scores[key][:, 1] = suppress * label_scores[key][:, 1]
#         label_scores[key][:, 2] = suppress * label_scores[key][:, 2]
#         label_scores[key][:, 3] = suppress * label_scores[key][:, 3]
#         label_scores[key][:, 0] = suppress * label_scores[key][:, 0]
#         label_scores[key][:, 5] = suppress * label_scores[key][:, 5]
#         label_scores[key][:, 6] = suppress * label_scores[key][:, 6]
#     elif key == 'Sad':
#         label_scores[key][:, 5] = label_weight[5] * label_scores[key][:, 5]
#         label_scores[key][:, 1] = suppress * label_scores[key][:, 1]
#         label_scores[key][:, 2] = suppress * label_scores[key][:, 2]
#         label_scores[key][:, 3] = suppress * label_scores[key][:, 3]
#         label_scores[key][:, 4] = 0.01 * label_scores[key][:, 4]
#         label_scores[key][:, 0] = suppress * label_scores[key][:, 0]
#         label_scores[key][:, 6] = suppress * label_scores[key][:, 6]
#     elif key == 'Surprise':
#         label_scores[key][:, 6] = label_weight[6] * label_scores[key][:, 6]
#         label_scores[key][:, 1] = suppress * label_scores[key][:, 1]
#         label_scores[key][:, 2] = suppress * label_scores[key][:, 2]
#         label_scores[key][:, 3] = suppress * label_scores[key][:, 3]
#         label_scores[key][:, 4] = 0.01 * label_scores[key][:, 4]
#         label_scores[key][:, 5] = suppress * label_scores[key][:, 5]
#         label_scores[key][:, 0] = suppress * label_scores[key][:, 0]
#     scores.append(label_scores[key])

score1 = label_scores['Angry']
score2 = label_scores['Disgust']
score3 = label_scores['Fear']
score4 = label_scores['Happy']
score5 = label_scores['Neutral']
score6 = label_scores['Sad']
score7 = label_scores['Surprise']
score8 = label_scores['overall']

for i in xrange(len(score1)):
    if i < 107:
        score1[i,:] = 3 * score1[i, :]
    elif i >= 107 and i < 136:
        score1[i,:] = 0.1 * score1[i, :]
    elif i >= 136 and i < 197:
        score1[i, :] = 0.1 * score1[i, :]
    elif i >= 197 and i < 313:
        score1[i, :] = 0.1 * score1[i, :]
    elif i >= 313 and i < 538:
        score1[i, :] = 0.1 * score1[i, :]
    elif i >= 538 and i < 606:
        score1[i, :] = 0.1 * score1[i, :]
    elif i >= 606 and i < 652:
        score1[i, :] = 0.1 * score1[i, :]

for i in xrange(len(score2)):
    if i < 107:
        score2[i,:] = 0.1 * score2[i, :]
    elif i >= 107 and i < 136:
        score2[i,:] = 3 * score2[i, :]
    elif i >= 136 and i < 197:
        score2[i, :] = 2 * score2[i, :]
    elif i >= 197 and i < 313:
        score2[i, :] = 0.1 * score2[i, :]
    elif i >= 313 and i < 538:
        score2[i, :] = 0.1 * score2[i, :]
    elif i >= 538 and i < 606:
        score2[i, :] = 0.1 * score2[i, :]
    elif i >= 606 and i < 652:
        score2[i, :] = 0.1 * score2[i, :]

for i in xrange(len(score3)):
    if i < 107:
        score3[i,:] = 0.1 * score3[i, :]
    elif i >= 107 and i < 136:
        score3[i,:] = 0.1 * score3[i, :]
    elif i >= 136 and i < 197:
        score3[i, :] = 3 * score3[i, :]
    elif i >= 197 and i < 313:
        score3[i, :] = 0.1 * score3[i, :]
    elif i >= 313 and i < 538:
        score3[i, :] = 0.1 * score3[i, :]
    elif i >= 538 and i < 606:
        score3[i, :] = 0.1 * score3[i, :]
    elif i >= 606 and i < 652:
        score3[i, :] = 0.1 * score3[i, :]

for i in xrange(len(score4)):
    if i < 107:
        score4[i,:] = 0.1 * score4[i, :]
    elif i >= 107 and i < 136:
        score4[i,:] = 0.1 * score4[i, :]
    elif i >= 136 and i < 197:
        score4[i, :] = 0.1 * score4[i, :]
    elif i >= 197 and i < 313:
        score4[i, :] = 3 * score4[i, :]
    elif i >= 313 and i < 538:
        score4[i, :] = 0.1 * score4[i, :]
    elif i >= 538 and i < 606:
        score4[i, :] = 0.1 * score4[i, :]
    elif i >= 606 and i < 652:
        score4[i, :] = 0.1 * score4[i, :]

for i in xrange(len(score5)):
    if i < 107:
        score5[i,:] = 0.1 * score5[i, :]
    elif i >= 107 and i < 136:
        score5[i,:] = 0.1 * score5[i, :]
    elif i >= 136 and i < 197:
        score5[i, :] = 0.1 * score5[i, :]
    elif i >= 197 and i < 313:
        score5[i, :] = 0.1 * score5[i, :]
    elif i >= 313 and i < 538:
        score5[i, :] = 3 * score5[i, :]
    elif i >= 538 and i < 606:
        score5[i, :] = 0.1 * score5[i, :]
    elif i >= 606 and i < 652:
        score5[i, :] = 0.1 * score5[i, :]

for i in xrange(len(score6)):
    if i < 107:
        score6[i,:] = 0.1 * score6[i, :]
    elif i >= 107 and i < 136:
        score6[i,:] = 0.1 * score6[i, :]
    elif i >= 136 and i < 197:
        score6[i, :] = 0.1 * score6[i, :]
    elif i >= 197 and i < 313:
        score6[i, :] = 0.1 * score6[i, :]
    elif i >= 313 and i < 538:
        score6[i, :] = 0.1 * score6[i, :]
    elif i >= 538 and i < 606:
        score6[i, :] = 3 * score6[i, :]
    elif i >= 606 and i < 652:
        score6[i, :] = 0.1 * score6[i, :]

for i in xrange(len(score7)):
    if i < 107:
        score7[i,:] = 0.1 * score7[i, :]
    elif i >= 107 and i < 136:
        score7[i,:] = 0.1 * score7[i, :]
    elif i >= 136 and i < 197:
        score7[i, :] = 0.1 * score7[i, :]
    elif i >= 197 and i < 313:
        score7[i, :] = 0.1 * score7[i, :]
    elif i >= 313 and i < 538:
        score7[i, :] = 0.1 * score7[i, :]
    elif i >= 538 and i < 606:
        score7[i, :] = 0.1 * score7[i, :]
    elif i >= 606 and i < 652:
        score7[i, :] = 3 * score7[i, :]

for i in xrange(len(score8)):
    if i < 107:
        score8[i,:] = 2 * score8[i, :]
    elif i >= 107 and i < 136:
        score8[i,:] = 0.1 * score8[i, :]
    elif i >= 136 and i < 197:
        score8[i, :] = 0.1 * score8[i, :]
    elif i >= 197 and i < 313:
        score8[i, :] = 2 * score8[i, :]
    elif i >= 313 and i < 538:
        score8[i, :] = 2 * score8[i, :]
    elif i >= 538 and i < 606:
        score8[i, :] = 2 * score8[i, :]
    elif i >= 606 and i < 652:
        score8[i, :] = 1 * score8[i, :]

scores = [score1, score2, score3, score4, score5, score6, score7, score8]

print(label_weight)
best_acc = 0
best_pred = 0


# 4개의 score를 [j, k, l, m] index로 가중치를 주고 더함
since = time.time()
for j in xrange(1, iter):
    print("Current step : {}".format(j))
    for k in xrange(1, iter):
        for l in xrange(1, iter):
            for m in xrange(1, iter):
                for n in xrange(1, iter):
                    for o in xrange(1, iter):
                        for p in xrange(1, iter):
                            for q in xrange(1, iter):
                                weights = [j, k, l, m, n, o, p, q]
                                ens_score = 0
                                for i in xrange(len(weights)):
                                    ens_score += weights[i] * scores[i]

                                ens_pred = np.argmax(ens_score, axis=1)
                                ens_pred = np.expand_dims(ens_pred, axis=1)
                                correct = (ens_pred == label).astype(np.int)
                                ens_acc = np.sum(correct / label.shape[0])
                                if best_acc < ens_acc:
                                    best_acc = ens_acc
                                    best_weight = [j, k, l, m, n, o, p, q]
                                    best_score = ens_score
                                    best_pred = ens_pred
                                    print('Ensemble Acc : {}'.format(best_acc))
                                    print(best_weight)

elapsed_time = time.time() - since
print('Elapsed Time per epoch : {:.0f}m {:.0f}s'.format(
    elapsed_time // 60, elapsed_time % 60))

# confusion matrix 만드는 방법, 여기서 label은 true label
conf = confusion_matrix(label, best_pred)

# plot confusion matrix using heat map
index = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
df_cm = pd.DataFrame(conf, index=index, columns=index)
plt.figure(figsize=(10, 7))
sn.heatmap(df_cm, annot=True, fmt='d')

f = open(SAVE + save_dir + '/config.txt', 'a')
f.write('Weight : ' + str(best_weight) + '\n')
plt.savefig(SAVE + save_dir + '/confusion.png')
np.savez(SAVE + save_dir + '/ens_scores.npz', best_score)
f.write('Acc : {:.2f}\n'.format(best_acc * 100))
f.close()

print("File name : " + save_dir)