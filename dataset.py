import os
import numpy as np
import random
import chainer

import utils as U




class SoundDataset(chainer.dataset.DatasetMixin):
    def __init__(self, sounds, labels, opt, train=True, elevations=None):
        if elevations is not None:
            self.base = chainer.datasets.TupleDataset(sounds, labels, elevations)
            self.has_elevation = True
        else:
            self.base = chainer.datasets.TupleDataset(sounds, labels)
            self.has_elevation = False
        self.opt = opt
        self.train = train
        self.mix = (opt.BC and train)
        self.preprocess_funcs = self.preprocess_setup()

    def __len__(self):
        return len(self.base)

    def preprocess_setup(self):
        if self.train:
            funcs = []
            if self.opt.strongAugment:
                funcs += [U.random_scale(1.25)]

            funcs += [U.padding(self.opt.inputLength // 2),
                      U.random_crop(self.opt.inputLength),
                      U.normalize(32768.0),
                      ]

        else:
            funcs = [U.padding(self.opt.inputLength // 2),
                     U.normalize(32768.0),
                     U.multi_crop(self.opt.inputLength, self.opt.nCrops),
                     ]

        return funcs

    def preprocess(self, sound):
        if sound.ndim == 2 and sound.shape[0] < sound.shape[1]:
            sound = sound.T
        for f in self.preprocess_funcs:
            sound = f(sound).astype(np.float32)
            #sound = f(sound).astype(np.float16)

        return sound
        #return sound

    def get_example(self, i):
        if self.mix:  # Training phase of BC learning
            # Select two training examples
            while True:
                sound1, label1 = self.base[random.randint(0, len(self.base) - 1)]
                sound2, label2 = self.base[random.randint(0, len(self.base) - 1)]
                #print(
                    #f"label1: {label1}, label2: {label2}, int(label1): {int(label1.split('.')[0])}, int(label2): {int(label2.split('.')[0])}")
                if label1 != label2:
                    break
            sound1 = self.preprocess(sound1)
            sound2 = self.preprocess(sound2)


            # Mix two examples
            r = np.array(random.random())
            sound = U.mix(sound1, sound2, r, self.opt.fs).astype(np.float32)
            #sound = U.mix(sound1, sound2, r, self.opt.fs).astype(np.float16)#replace float32 by float16 due to memory problem
            eye = np.eye(self.opt.nClasses)
            #label = (eye[int(label1)] * r + eye[int(label2)] * (1 - r)).astype(np.float32) #used for esc and urbansound8k
            label = (eye[int(label1)-1] * r + eye[int(label2)-1] * (1 - r)).astype(np.float32) #used for samrai,replace float32 by float16 due to memory problem
            return sound, label


        else:# for standard training or test
            elems = self.base[i]
            sound, label = elems[:2]
            sound = self.preprocess(sound).astype(np.float32)
            label = np.array(label, dtype=np.int32)

            if self.train and self.opt.strongAugment:
                sound = U.random_gain(6)(sound).astype(np.float32)

            if self.has_elevation:# if it has elevations
                elevation = elems[2]
                return sound, label, elevation
            else:
                return sound, label


def setup(opt, split):
    dataset = np.load(os.path.join(opt.data, opt.dataset, 'wav{}.npz'.format(opt.fs // 1000)), allow_pickle=True)
    #print("Loaded dataset keys:", dataset.keys())
    #dataset_path = os.path.join(opt.data, opt.dataset, 'wav{}.npz'.format(opt.fs // 1000))
    #dataset = np.load(dataset_path, mmap_mode='r', allow_pickle=True)

    # Split to train and val
    train_sounds = []
    train_labels = []
    val_sounds = []
    val_labels = []
    val_elevations = []
    if opt.dataset.startswith("samrai_test_"): # setting for test datasets
        for i in range(5, opt.nFolds + 5):
            sounds = dataset['fold{}'.format(i)].item()['sounds']
            labels = dataset['fold{}'.format(i)].item()['labels']
            elevations = dataset['fold{}'.format(i)].item()['elevations']
            # print('Successfully loaded elevation information')
            if i == split+4:
                val_sounds.extend(sounds)
                val_labels.extend(labels)
                val_elevations.extend(elevations)
                # print('Successfully extended elevation information')
            else:
                train_sounds.extend(sounds)
                train_labels.extend(labels)
    else:
        for i in range(1, opt.nFolds + 1):
            sounds = dataset['fold{}'.format(i)].item()['sounds']
            labels = dataset['fold{}'.format(i)].item()['labels']
            if i == split:
                val_sounds.extend(sounds)
                val_labels.extend(labels)
            else:
                train_sounds.extend(sounds)
                train_labels.extend(labels)


    # Free memory after splitting
    del dataset
    # print("Training sounds count:", len(train_sounds))
    # print("Validation sounds count:", len(val_sounds))

    # Iterator setup
    train_data = SoundDataset(train_sounds, train_labels, opt, train=True)
    if opt.dataset.startswith("samrai_test_"):  # setting for test datasets
        val_data = SoundDataset(val_sounds, val_labels, opt, train=False, elevations=val_elevations)
    else:
        val_data = SoundDataset(val_sounds, val_labels, opt, train=False)
    if opt.netType == 'envnetstero':
        train_iter = chainer.iterators.MultiprocessIterator(train_data, opt.batchSize, repeat=False,
                                                            dataset_timeout=None, n_processes=12)
    elif opt.netType == 'envnetsterov2':
        train_iter = chainer.iterators.MultiprocessIterator(train_data, opt.batchSize, repeat=False,
                                                            dataset_timeout=None, n_processes=2)
    else:
        train_iter = chainer.iterators.MultiprocessIterator(train_data, opt.batchSize, repeat=False)
    #print(f"Number of processes used: {train_iter.n_processes}")
    val_iter = chainer.iterators.SerialIterator(val_data, opt.batchSize // opt.nCrops, repeat=False, shuffle=False)

    return train_iter, val_iter
