import os
import argparse


def parse():
    parser = argparse.ArgumentParser(description='BC learning for sounds')

    # General settings
    parser.add_argument('--dataset', required=True, choices=['esc10', 'esc50', 'samrai','samrai_original','samrai_test_original','urbansound8k','samrai_arma','samrai_averaged','samrai_N1N2P1P2','samrai_test_arma','samrai_test_averaged','samrai_test_N1N2P1P2'])
    parser.add_argument('--netType', required=True, choices=['envnet', 'envnetstereo','envnetv2', 'envnetstereov2', 'envnetstereov2_1'])
    parser.add_argument('--data', required=True, help='Path to dataset')
    parser.add_argument('--split', type=int, default=-1, help='esc: 1-5, samrai: 1-4, samrai_original: 1-4, samrai_test_original: 5, urbansound: 1-10 (-1: run on all splits), samrai_arma: 1-4, samrai_averaged: 1-4, samrai_N1N2P1P2: 1-4, samrai_test_arma: 5, samrai_test_averaged: 5, samrai_test_N1N2P1P2: 5')
    parser.add_argument('--save', default='None', help='Directory to save the results')
    parser.add_argument('--testOnly', action='store_true')
    parser.add_argument('--gpu', type=int, default=0)

    # Learning settings (default settings are defined below)
    parser.add_argument('--BC', action='store_true', help='BC learning')
    parser.add_argument('--strongAugment', action='store_true', help='Add scale and gain augmentation')
    parser.add_argument('--nEpochs', type=int, default=-1)
    parser.add_argument('--LR', type=float, default=-1, help='Initial learning rate')
    parser.add_argument('--schedule', type=float, nargs='*', default=-1, help='When to divide the LR')
    parser.add_argument('--warmup', type=int, default=-1, help='Number of epochs to warm up')
    parser.add_argument('--batchSize', type=int, default=64)
    parser.add_argument('--weightDecay', type=float, default=5e-4)
    parser.add_argument('--momentum', type=float, default=0.9)

    # Testing settings
    parser.add_argument('--nCrops', type=int, default=10)

    opt = parser.parse_args()

    # Dataset details
    if opt.dataset == 'esc50':
        opt.nClasses = 50
        opt.nFolds = 5
    elif opt.dataset == 'esc10':
        opt.nClasses = 10
        opt.nFolds = 5
    elif opt.dataset == 'samrai':
        opt.nClasses = 596
        opt.nFolds = 4
    elif opt.dataset == 'samrai_original':
        opt.nClasses = 11
        opt.nFolds = 4
    elif opt.dataset == 'samrai_test_original':
        opt.nClasses = 11
        opt.nFolds = 1
    elif opt.dataset == 'samrai_arma':
        opt.nClasses = 11
        opt.nFolds = 4
    elif opt.dataset == 'samrai_averaged':
        opt.nClasses = 11
        opt.nFolds = 4
    elif opt.dataset == 'samrai_N1N2P1P2':
        opt.nClasses = 11
        opt.nFolds = 4
    elif opt.dataset == 'samrai_test_arma':
        opt.nClasses = 11
        opt.nFolds = 1
    elif opt.dataset == 'samrai_test_averaged':
        opt.nClasses = 11
        opt.nFolds = 1
    elif opt.dataset == 'samrai_test_N1N2P1P2':
        opt.nClasses = 11
        opt.nFolds = 1
    else:  # urbansound8k
        opt.nClasses = 10
        opt.nFolds = 10

    if opt.split == -1:
        opt.splits = range(1, opt.nFolds + 1)
    else:
        opt.splits = [opt.split]

    # Model details
    if opt.netType == 'envnet':
        opt.fs = 16000
        opt.inputLength = 24014
    elif opt.netType == 'envnetstereo':
        opt.fs = 16000
        opt.inputLength = 17000
    elif opt.netType == 'envnetstereov2':
        opt.fs = 44100
        opt.inputLength = 47000
    elif opt.netType == 'envnetstereov2_1':
        opt.fs = 44100
        opt.inputLength = 47000
    else:  # envnetv2
        opt.fs = 44100
        opt.inputLength = 66650

    # Default settings (nEpochs will be doubled if opt.BC)
    default_settings = dict()
    default_settings['esc50'] = {
        'envnet': {'nEpochs': 600, 'LR': 0.01, 'schedule': [0.5, 0.75], 'warmup': 0},
        'envnetv2': {'nEpochs': 1000, 'LR': 0.1, 'schedule': [0.3, 0.6, 0.9], 'warmup': 10}
    }
    default_settings['esc10'] = {
        'envnet': {'nEpochs': 600, 'LR': 0.01, 'schedule': [0.5, 0.75], 'warmup': 0},
        #'envnet': {'nEpochs': 600, 'LR': 0.01, 'schedule': [0.75], 'warmup': 0},
        'envnetv2': {'nEpochs': 600, 'LR': 0.01, 'schedule': [0.5, 0.75], 'warmup': 0}
    }
    default_settings['samrai'] = {
        'envnet': {'nEpochs': 600, 'LR': 0.01, 'schedule': [0.5, 0.75], 'warmup': 0},
        'envnetv2': {'nEpochs': 600, 'LR': 0.01, 'schedule': [0.3, 0.6, 0.9], 'warmup': 10}
    }
    default_settings['samrai_original'] = {
        'envnet': {'nEpochs': 600, 'LR': 0.01, 'schedule': [0.5, 0.75], 'warmup': 0},
        'envnetstereo': {'nEpochs': 400, 'LR': 0.01, 'schedule': [0.75], 'warmup': 0},
        'envnetv2': {'nEpochs': 600, 'LR': 0.01, 'schedule': [0.3, 0.6, 0.9], 'warmup': 10},
        'envnetstereov2': {'nEpochs': 150, 'LR': 0.01, 'schedule': [0.6, 0.9], 'warmup': 10},
        'envnetstereov2_1': {'nEpochs': 150, 'LR': 0.01, 'schedule': [0.6, 0.9], 'warmup': 10}
        #'envnetstereov2': {'nEpochs': 400, 'LR': 0.01, 'schedule': [0.6, 0.9], 'warmup': 10}
    }
    default_settings['samrai_arma'] = {
        'envnet': {'nEpochs': 600, 'LR': 0.01, 'schedule': [0.5, 0.75], 'warmup': 0},
        'envnetstereo': {'nEpochs': 400, 'LR': 0.01, 'schedule': [0.75], 'warmup': 0},
        'envnetv2': {'nEpochs': 600, 'LR': 0.01, 'schedule': [0.3, 0.6, 0.9], 'warmup': 10},
        'envnetstereov2': {'nEpochs': 150, 'LR': 0.01, 'schedule': [0.6, 0.9], 'warmup': 10},
        'envnetstereov2_1': {'nEpochs': 150, 'LR': 0.01, 'schedule': [0.6, 0.9], 'warmup': 10}
        # 'envnetstereov2': {'nEpochs': 400, 'LR': 0.01, 'schedule': [0.6, 0.9], 'warmup': 10}
    }
    default_settings['samrai_averaged'] = {
        'envnet': {'nEpochs': 600, 'LR': 0.01, 'schedule': [0.5, 0.75], 'warmup': 0},
        'envnetstereo': {'nEpochs': 400, 'LR': 0.01, 'schedule': [0.75], 'warmup': 0},
        'envnetv2': {'nEpochs': 600, 'LR': 0.01, 'schedule': [0.3, 0.6, 0.9], 'warmup': 10},
        'envnetstereov2': {'nEpochs': 150, 'LR': 0.01, 'schedule': [0.6, 0.9], 'warmup': 10},
        'envnetstereov2_1': {'nEpochs': 150, 'LR': 0.01, 'schedule': [0.6, 0.9], 'warmup': 10}
        # 'envnetstereov2': {'nEpochs': 400, 'LR': 0.01, 'schedule': [0.6, 0.9], 'warmup': 10}
    }
    default_settings['samrai_N1N2P1P2'] = {
        'envnet': {'nEpochs': 600, 'LR': 0.01, 'schedule': [0.5, 0.75], 'warmup': 0},
        'envnetstereo': {'nEpochs': 400, 'LR': 0.01, 'schedule': [0.75], 'warmup': 0},
        'envnetv2': {'nEpochs': 600, 'LR': 0.01, 'schedule': [0.3, 0.6, 0.9], 'warmup': 10},
        'envnetstereov2': {'nEpochs': 150, 'LR': 0.01, 'schedule': [0.6, 0.9], 'warmup': 10},
        'envnetstereov2_1': {'nEpochs': 150, 'LR': 0.01, 'schedule': [0.6, 0.9], 'warmup': 10}
        # 'envnetstereov2': {'nEpochs': 400, 'LR': 0.01, 'schedule': [0.6, 0.9], 'warmup': 10}
    }
    default_settings['samrai_test_original'] = {
        'envnet': {'nEpochs': 600, 'LR': 0.01, 'schedule': [0.5, 0.75], 'warmup': 0},
        'envnetstereo': {'nEpochs': 400, 'LR': 0.01, 'schedule': [0.75], 'warmup': 0},
        'envnetv2': {'nEpochs': 600, 'LR': 0.01, 'schedule': [0.3, 0.6, 0.9], 'warmup': 10},
        'envnetstereov2': {'nEpochs': 150, 'LR': 0.01, 'schedule': [0.6, 0.9], 'warmup': 10},
        'envnetstereov2_1': {'nEpochs': 150, 'LR': 0.01, 'schedule': [0.6, 0.9], 'warmup': 10}
    }
    default_settings['samrai_test_arma'] = {
        'envnet': {'nEpochs': 600, 'LR': 0.01, 'schedule': [0.5, 0.75], 'warmup': 0},
        'envnetstereo': {'nEpochs': 400, 'LR': 0.01, 'schedule': [0.75], 'warmup': 0},
        'envnetv2': {'nEpochs': 600, 'LR': 0.01, 'schedule': [0.3, 0.6, 0.9], 'warmup': 10},
        'envnetstereov2': {'nEpochs': 150, 'LR': 0.01, 'schedule': [0.6, 0.9], 'warmup': 10},
        'envnetstereov2_1': {'nEpochs': 150, 'LR': 0.01, 'schedule': [0.6, 0.9], 'warmup': 10}
    }
    default_settings['samrai_test_averaged'] = {
        'envnet': {'nEpochs': 600, 'LR': 0.01, 'schedule': [0.5, 0.75], 'warmup': 0},
        'envnetstereo': {'nEpochs': 400, 'LR': 0.01, 'schedule': [0.75], 'warmup': 0},
        'envnetv2': {'nEpochs': 600, 'LR': 0.01, 'schedule': [0.3, 0.6, 0.9], 'warmup': 10},
        'envnetstereov2': {'nEpochs': 150, 'LR': 0.01, 'schedule': [0.6, 0.9], 'warmup': 10},
        'envnetstereov2_1': {'nEpochs': 150, 'LR': 0.01, 'schedule': [0.6, 0.9], 'warmup': 10}
    }
    default_settings['samrai_test_N1N2P1P2'] = {
        'envnet': {'nEpochs': 600, 'LR': 0.01, 'schedule': [0.5, 0.75], 'warmup': 0},
        'envnetstereo': {'nEpochs': 400, 'LR': 0.01, 'schedule': [0.75], 'warmup': 0},
        'envnetv2': {'nEpochs': 600, 'LR': 0.01, 'schedule': [0.3, 0.6, 0.9], 'warmup': 10},
        'envnetstereov2': {'nEpochs': 150, 'LR': 0.01, 'schedule': [0.6, 0.9], 'warmup': 10},
        'envnetstereov2_1': {'nEpochs': 150, 'LR': 0.01, 'schedule': [0.6, 0.9], 'warmup': 10}
    }
    default_settings['urbansound8k'] = {
        'envnet': {'nEpochs': 400, 'LR': 0.01, 'schedule': [0.5, 0.75], 'warmup': 0},
        'envnetv2': {'nEpochs': 600, 'LR': 0.1, 'schedule': [0.3, 0.6, 0.9], 'warmup': 10}
    }
    for key in ['nEpochs', 'LR', 'schedule', 'warmup']:
        if eval('opt.{}'.format(key)) == -1:
            setattr(opt, key, default_settings[opt.dataset][opt.netType][key])
            if key == 'nEpochs' and opt.BC:
                opt.nEpochs *= 2

    if opt.save != 'None' and not os.path.isdir(opt.save):
        os.makedirs(opt.save)

    display_info(opt)

    return opt


def display_info(opt):
    if opt.BC:
        learning = 'BC'
    else:
        learning = 'standard'

    print('+------------------------------+')
    print('| Sound classification')
    print('+------------------------------+')
    print('| dataset  : {}'.format(opt.dataset))
    print('| netType  : {}'.format(opt.netType))
    print('| learning : {}'.format(learning))
    print('| augment  : {}'.format(opt.strongAugment))
    print('| nEpochs  : {}'.format(opt.nEpochs))
    print('| LRInit   : {}'.format(opt.LR))
    print('| schedule : {}'.format(opt.schedule))
    print('| warmup   : {}'.format(opt.warmup))
    print('| batchSize: {}'.format(opt.batchSize))
    print('+------------------------------+')
