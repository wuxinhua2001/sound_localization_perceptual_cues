import sys
import os
import subprocess

import glob
import numpy as np
import wavio

def main():
    samrai_path = os.path.join(sys.argv[1], 'samrai')#4 folders: fold1-fold4
    samrai_stereo_path = os.path.join(sys.argv[1], 'samrai_stereo')#4 folders: fold1-fold4

    samrai_test_original_path = os.path.join(sys.argv[1], 'samrai_test_original')#1 folder: fold5
    samrai_test_arma_path = os.path.join(sys.argv[1], 'samrai_test_arma')
    samrai_test_averaged_path = os.path.join(sys.argv[1], 'samrai_test_averaged')
    samrai_test_N1N2P1P2_path = os.path.join(sys.argv[1], 'samrai_test_N1N2P1P2')

    samrai_test_original_metadata_path = os.path.join(sys.argv[1], 'metadata_test_SAMRAI_original')
    samrai_test_arma_metadata_path = os.path.join(sys.argv[1], 'metadata_test_SAMRAI_AR10MA10')
    samrai_test_averaged_metadata_path = os.path.join(sys.argv[1], 'metadata_test_SAMRAI_averaged')
    samrai_test_N1N2P1P2_metadata_path = os.path.join(sys.argv[1], 'metadata_test_SAMRAI_N1N2P1P2')


    #os.mkdir(samrai_path)
    fs_list = [16000, 44100]

    # Convert sampling rate
    # for fs in fs_list:
    #     convert_fs(samrai_path,
    #                os.path.join(samrai_stereo_path, 'wav{}'.format(fs // 1000)),
    #                fs)

    for fs in fs_list:
        # src_path = os.path.join(samrai_test_averaged_path, 'wav{}'.format(fs // 1000))
        # create_dataset(src_path, src_path + '.npz', samrai_test_averaged_metadata_path)
        src_path = os.path.join(samrai_test_N1N2P1P2_path, 'wav{}'.format(fs // 1000))
        create_dataset(src_path, src_path + '.npz', samrai_test_N1N2P1P2_metadata_path)

def convert_fs(src_path, dst_path, fs):
    print('* {} -> {}'.format(src_path, dst_path))
    os.mkdir(dst_path)
    for src_file in sorted(glob.glob(os.path.join(src_path, '*.wav'))):
        dst_file = src_file.replace(src_path, dst_path)
        #subprocess.call('ffmpeg -i {} -ac 1 -ar {} -loglevel error -y {}'.format(src_file, fs, dst_file), shell=True)
        # Remove '-ac 1' to keep the original channels (stereo)
        subprocess.call('ffmpeg -i {} -ar {} -loglevel error -y {}'.format(src_file, fs, dst_file), shell=True)

def create_dataset(src_path, dst_path, csv_path):
    print('* {} -> {}'.format(src_path, dst_path))
    dataset = {}

    # correspondance of elevation angle
    angle_map = {
        (0, 0): 0,
        (0, 30): 30,
        (0, 60): 60,
        (0, 90): 90,
        (180, 60): 120,
        (180, 30): 150,
        (180, 0): 180
    }

    for fold in range(5, 6): # range(1,5) for training datasets, range(5,6) for test datasets
        dataset['fold{}'.format(fold)] = {}
        sounds = []
        labels = []
        elevations = []  # 新增仰角列表

        for wav_file in sorted(glob.glob(os.path.join(src_path, 'fold{}_*.wav'.format(fold)))):
            #sound = wavio.read(wav_file).data.T[0]
            sound = wavio.read(wav_file).data.T# Keep both channels
            #print(os.path.splitext(wav_file)[0].split('_'))
            file_name = os.path.basename(wav_file)
            #print(file_name)
            label = int(os.path.splitext(file_name)[0].split('_')[1])
            #label = int(os.path.splitext(file_name)[0].split('_')[2])
            sounds.append(sound)
            labels.append(label)

            # ===== 仰角提取逻辑 =====
            # 解析出与CSV文件关联的后缀编号
            file_id = os.path.splitext(file_name)[0].split('_')[-1]
            csv_name = f'fold{fold}_{file_id}.csv'
            csv_file_path = os.path.join(csv_path, csv_name)

            try:
                with open(csv_file_path, 'r') as f:
                    first_line = f.readline().strip()
                    values = [int(x) for x in first_line.split(',')]
                    az, el = values[-2], values[-1]
                    elevation = angle_map.get((az, el), -1)  # 默认-1表示未识别
            except Exception as e:
                print(f'Warning: Failed to read angle from {csv_path}, defaulting to -1. Error: {e}')
                elevation = -1

            elevations.append(elevation)

        dataset['fold{}'.format(fold)]['sounds'] = sounds
        dataset['fold{}'.format(fold)]['labels'] = labels
        dataset['fold{}'.format(fold)]['elevations'] = elevations  # 添加仰角信息

    np.savez(dst_path, **dataset)

if __name__ == '__main__':
    main()
