import numpy as np
import random
import chainer.functions as F
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import glob
import re


# Default data augmentation
def padding(pad):
    def f(sound):
        if sound.ndim == 2:  # binaural
            return np.pad(sound, ((pad, pad), (0, 0)), 'constant')
        else:  # monaural
            return np.pad(sound, pad, 'constant')

    return f



def random_crop(size):
    def f(sound):
        org_size = len(sound)  # 时间轴长度
        #print("the signal length before cropping:",len(sound))
        start = random.randint(0, org_size - size)
        if sound.ndim == 2:  # binaural
            return sound[start: start + size,:]  # crop the time sequence
        else:  # monaural
            return sound[start: start + size]

    return f



def normalize(factor):
    def f(sound):
        #return sound / factor
        return (sound / factor).astype(np.float32)

    return f


# For strong data augmentation
def random_scale(max_scale, interpolate='Linear'):
    def f(sound):
        scale = np.power(max_scale, random.uniform(-1, 1))
        output_size = int(len(sound) * scale)
        ref = np.arange(output_size) / scale
        if interpolate == 'Linear':
            ref1 = ref.astype(np.int32)
            #ref1 = ref.astype(np.int16)
            ref2 = np.minimum(ref1 + 1, len(sound) - 1)
            r = ref - ref1
            scaled_sound = sound[ref1] * (1 - r) + sound[ref2] * r
        elif interpolate == 'Nearest':
            scaled_sound = sound[ref.astype(np.int32)]
            #scaled_sound = sound[ref.astype(np.int16)]
        else:
            raise Exception('Invalid interpolation mode {}'.format(interpolate))

        return scaled_sound

    return f


def random_gain(db):
    def f(sound):
        return sound * np.power(10, random.uniform(-db, db) / 20.0)

    return f


# For testing phase
def multi_crop(input_length, n_crops):
    def f(sound):
        stride = (len(sound) - input_length) // (n_crops - 1)
        # stride = (len(sound) - input_length) // n_crops # test for ncrops = 1

        # process for different sound dimension
        if sound.ndim == 2:  # 双声道信号
            sounds = [sound[stride * i: stride * i + input_length, :] for i in range(n_crops)]
        else:  # 单声道信号
            sounds = [sound[stride * i: stride * i + input_length] for i in range(n_crops)]

        return np.array(sounds)

    return f


# For BC learning
def a_weight(fs, n_fft, min_db=-80.0):
    freq = np.linspace(0, fs // 2, n_fft // 2 + 1)
    freq_sq = np.power(freq, 2)
    freq_sq[0] = 1.0
    weight = 2.0 + 20.0 * (2 * np.log10(12194) + 2 * np.log10(freq_sq)
                           - np.log10(freq_sq + 12194 ** 2)
                           - np.log10(freq_sq + 20.6 ** 2)
                           - 0.5 * np.log10(freq_sq + 107.7 ** 2)
                           - 0.5 * np.log10(freq_sq + 737.9 ** 2))
    weight = np.maximum(weight, min_db)

    return weight


def compute_gain(sound, fs, min_db=-80.0, mode='A_weighting'):
    if fs == 16000:
        n_fft = 2048
    elif fs == 44100:
        n_fft = 4096
    else:
        raise Exception('Invalid fs {}'.format(fs))
    stride = n_fft // 2

    gain = []
    for i in range(0, len(sound) - n_fft + 1, stride):
        if mode == 'RMSE':
            g = np.mean(sound[i: i + n_fft] ** 2)
        elif mode == 'A_weighting':
            spec = np.fft.rfft(np.hanning(n_fft + 1)[:-1] * sound[i: i + n_fft])
            power_spec = np.abs(spec) ** 2
            a_weighted_spec = power_spec * np.power(10, a_weight(fs, n_fft) / 10)
            g = np.sum(a_weighted_spec)
        else:
            raise Exception('Invalid mode {}'.format(mode))
        gain.append(g)

    gain = np.array(gain)
    gain = np.maximum(gain, np.power(10, min_db / 10))
    gain_db = 10 * np.log10(gain)

    return gain_db


def mix(sound1, sound2, r, fs):
    gain1 = np.max(compute_gain(sound1, fs))  # Decibel
    gain2 = np.max(compute_gain(sound2, fs))
    t = 1.0 / (1 + np.power(10, (gain1 - gain2) / 20.) * (1 - r) / r)
    sound = ((sound1 * t + sound2 * (1 - t)) / np.sqrt(t ** 2 + (1 - t) ** 2))

    return sound


def kl_divergence(y, t):
    entropy = - F.sum(t[t.data.nonzero()] * F.log(t[t.data.nonzero()]))
    crossEntropy = - F.sum(t * F.log_softmax(y))

    return (crossEntropy - entropy) / y.shape[0]


# Convert time representation
def to_hms(time):
    h = int(time // 3600)
    m = int((time - h * 3600) // 60)
    s = int(time - h * 3600 - m * 60)
    if h > 0:
        line = '{}h{:02d}m'.format(h, m)
    else:
        line = '{}m{:02d}s'.format(m, s)

    return line

def draw_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    #labels = ["knock", "drawer", "clean throat", "phone", "keys drop", "speech", "keyboard", "page turn", "cough", "door slam", "laughter"]# 适用于 11 类
    labels = np.arange(1,597)

    # 绘制混淆矩阵
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    plt.show()

def extract_angles_for_predictions(y_true, y_pred, i_hrir_to_angle):
    # 初始化列表
    y_true_azi = []
    y_true_ele = []
    y_pred_azi = []
    y_pred_ele = []

    # 遍历真实值和预测值列表
    for true, pred in zip(y_true, y_pred):
        if true in i_hrir_to_angle:
            true_azi, true_ele = i_hrir_to_angle[true]
            y_true_azi.append(true_azi)
            y_true_ele.append(true_ele)
        else:
            print(f"Warning: true value of i_hrir {true} not found in i_hrir_to_angle")

        if pred in i_hrir_to_angle:
            pred_azi, pred_ele = i_hrir_to_angle[pred]
            y_pred_azi.append(pred_azi)
            y_pred_ele.append(pred_ele)
        else:
            print(f"Warning: predicted value of i_hrir {pred} not found in i_hrir_to_angle")

    return y_true_azi, y_true_ele, y_pred_azi, y_pred_ele

def match_doa_to_angles():
    # set the path to the folder of csv files
    csv_folder = "/home/wu/datasets/metadata_samrai_stereo/"
    csv_files = glob.glob(csv_folder + "*.csv")  # get all the csv files
    #print(f"csv files: {len(csv_files)}")
    # 用字典存储 i_hrir → (azimuth, elevation)
    i_hrir_to_angle = {}
    for csv_file in csv_files:
        with open(csv_file, "r") as f:
            first_line = next(f).strip()  # 读取第一行
            parts = first_line.split(",")  # 按逗号拆分
            #print(parts)
            filename = parts[-1]  # 文件名在最后一列
            match = re.search(r'_(\d+)_(\d+)\.wav$', filename)
            if match:
                i_hrir = int(match.group(1))
                #print(i_hrir)
                azimuth = float(parts[3])  # 第 4 列是 azimuth
                #print(azimuth)
                elevation = float(parts[4])  # 第 5 列是 elevation
                #print(elevation)

                if i_hrir not in i_hrir_to_angle:
                    i_hrir_to_angle[i_hrir] = (azimuth, elevation)
    return i_hrir_to_angle

def save_split_result(split, train_losses, train_tops, val_tops):
    """save the result after each split"""

    # create the saving directory
    result_dir = f'./plots/results_split_{split}'
    os.makedirs(result_dir, exist_ok=True)

    # draw the training loss
    plt.figure()
    plt.plot(train_losses, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Training Loss for Split {split}')
    plt.legend()
    plt.savefig(os.path.join(result_dir, f'split_{split}_training_loss.png'))
    plt.close()

    # draw the training and validation error rate
    plt.figure()
    plt.plot(train_tops, label='Training Error Rate', color='blue', linestyle='-', marker='o')
    plt.plot(val_tops, label='Validation Error Rate', color='red', linestyle='--', marker='x')
    plt.xlabel('Epoch')
    plt.ylabel('Error Rate(%)')
    plt.title(f'Training and Validation Error rate for Split {split}')
    plt.ylim(0, 100)
    plt.yticks(range(0, 101, 10))
    plt.legend()
    plt.savefig(os.path.join(result_dir, f'split_{split}_training_error_rate.png'))
    plt.close()

def plot_azi(y_true_azi, y_pred_azi):
    if len(y_true_azi) == len(y_pred_azi):
        # 绘制方位角 (Azimuth) 散点图
        plt.figure(figsize=(8, 6))
        plt.scatter(y_pred_azi, y_true_azi, alpha=0.5, label='Azimuth Scatter')
        plt.plot([-180, 180], [-180, 180], 'r--', label='Ideal Prediction')  # 理想预测的对角线
        plt.xlim(-180, 180)
        plt.ylim(-180, 180)
        plt.xlabel('Predicted Azimuth (°)')
        plt.ylabel('True Azimuth (°)')
        plt.title('Azimuth Prediction Scatter Plot')
        plt.legend()
        plt.grid(True)
        plt.show()
    else:
        print("Warning: Lengths of true and predicted azimuth angles are not equal.")


def plot_ele(y_true_ele, y_pred_ele):
    if len(y_true_ele) == len(y_pred_ele):
        # 绘制仰角 (Elevation) 散点图
        plt.figure(figsize=(8, 6))
        plt.scatter(y_pred_ele, y_true_ele, alpha=0.5, label='Elevation Scatter')
        plt.plot([-60, 90], [-60, 90], 'r--', label='Ideal Prediction')  # 理想预测的对角线
        plt.xlim(-60, 90)
        plt.ylim(-60, 90)
        plt.xlabel('Predicted Elevation (°)')
        plt.ylabel('True Elevation (°)')
        plt.title('Elevation Prediction Scatter Plot')
        plt.legend()
        plt.grid(True)
        plt.show()
    else:
        print("Warning: Lengths of true and predicted elevation angles are not equal.")
