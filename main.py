"""
 Learning from Between-class Examples for Deep Sound Recognition.
 Yuji Tokozume, Yoshitaka Ushiku, and Tatsuya Harada

"""

import sys
import os
import chainer
import matplotlib
#matplotlib.use('Agg')#configuration to avoid using tkinter
from chainer.backends import cuda #used for chainer7.8.1
from chainer import optimizer_hooks
import opts
import models
import dataset
import numpy as np
from train import Trainer
import utils as U

def main():
    cuda.set_max_workspace_size(1024 * 1024 * 500)  # increase the cudnn workspace size to 500MB
    opt = opts.parse()
    #chainer.cuda.get_device_from_id(opt.gpu).use()
    #cuda.get_device_from_id(opt.gpu).use()
    device = cuda.get_device_from_id(opt.gpu) #used for chainer7.8.1
    #mem_info = cuda.cupy.cuda.runtime.memGetInfo()
    #total_mem_gb = mem_info[1]/(1024**3)
    #available_mem_gb = mem_info[0]/(1024**3)
    #print(f"Total GPU memory: {total_mem_gb:.2f} GB")
    #print(f"Available GPU memory: {available_mem_gb:.2f} GB")
    with device:
        for split in opt.splits:
            print('+-- Split {} --+'.format(split))
            train(opt, split)  # use this line in testOnly mode and comment the two lines below
            #train_losses, train_tops, val_tops = train(opt, split)
            #U.save_split_result(split, train_losses, train_tops, val_tops) #uncomment this line if you want to draw the variation of training loss as well as training and validation error rate

def train(opt, split):
    model = getattr(models, opt.netType)(opt.nClasses)
    model.to_gpu()
    optimizer = chainer.optimizers.NesterovAG(lr=opt.LR, momentum=opt.momentum)
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer_hooks.WeightDecay(opt.weightDecay)) #replaced optimizer by optimizer_hooks
    train_iter, val_iter = dataset.setup(opt, split)
    trainer = Trainer(model, optimizer, train_iter, val_iter, opt)

    # train_losses = []
    # train_tops = []
    # val_tops = []

    if opt.testOnly:
        chainer.serializers.load_npz(
            os.path.join(opt.save, 'model_split{}.npz'.format(split)), trainer.model)
        val_top1, feature_data = trainer.val_testOnly()

        # save the output features of test
        save_path = os.path.join(opt.save, opt.dataset)
        os.makedirs(save_path, exist_ok=True)
        np.savez(os.path.join(save_path, "val_feature_data_split{}_new_v3.npz".format(split)), **feature_data) # save the output features of test

        print('| Val: top1 {:.2f}'.format(val_top1))

        # visualize the true and predicted value(currently unneeded)
        # y_pred = np.array([y.item() for y in y_pred])  # transfer to int
        # i_hrir_to_angle = U.match_doa_to_angles()
        # y_true_azi, y_true_ele, y_pred_azi, y_pred_ele = U.extract_angles_for_predictions(y_true,y_pred, i_hrir_to_angle)
        #print(len(y_true_azi))
        #print(len(y_pred_azi))
        #print(len(y_true_ele))
        #print(len(y_pred_ele))
        # U.plot_azi(y_true_azi, y_pred_azi)
        # U.plot_ele(y_true_ele, y_pred_ele)
        return

    for epoch in range(1, opt.nEpochs + 1):
        train_loss, train_top1 = trainer.train(epoch)
        # train_tops.append(train_top1)
        # train_losses.append(train_loss)
        val_top1 = trainer.val()
        #val_tops.append(val_top1)
        sys.stderr.write('\r\033[K')
        sys.stdout.write(
            '| Epoch: {}/{} | Train: LR {}  Loss {:.3f}  top1 {:.2f} | Val: top1 {:.2f}\n'.format(
                epoch, opt.nEpochs, trainer.optimizer.lr, train_loss, train_top1, val_top1))
        sys.stdout.flush()

    if opt.save != 'None':
        chainer.serializers.save_npz(
            os.path.join(opt.save, 'model_split{}.npz'.format(split)), model)

    #return train_losses, train_tops, val_tops

if __name__ == '__main__':
    main()

