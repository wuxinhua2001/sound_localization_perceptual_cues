import sys
import numpy as np
import chainer
#from chainer import cuda
from chainer.backends import cuda #used for chainer7.8.1
import chainer.functions as F
import time
# import cupy as cp

import utils
# Deprecated
# from feature_utils import initialize_feature_storage, record_feature_batch


class Trainer:
    def __init__(self, model, optimizer, train_iter, val_iter, opt):
        self.model = model
        self.optimizer = optimizer
        self.train_iter = train_iter
        self.val_iter = val_iter
        self.opt = opt
        self.n_batches = (len(train_iter.dataset) - 1) // opt.batchSize + 1
        self.start_time = time.time()

    def train(self, epoch):
        self.model.train = True
        with chainer.using_config('train', self.model.train):#configuration for chainer7.8.1
            self.optimizer.lr = self.lr_schedule(epoch)
            train_loss = 0
            train_acc = 0
            for i, batch in enumerate(self.train_iter):
                x_array, t_array = chainer.dataset.concat_examples(batch)
                #print(f"X array shape in training:{x_array.shape}") # shape: (batchSize, length, number of channels)
                if x_array.ndim == 3:
                    x_array = x_array.transpose(0,2,1)
                    x_array = x_array[:, :, None, :]
                else:
                    x_array = x_array[:, None, None, :]

                x = chainer.Variable(cuda.to_gpu(x_array))
                t = chainer.Variable(cuda.to_gpu(t_array))
                self.model.cleargrads()
                y = self.model(x)
                if self.opt.BC:
                    loss = utils.kl_divergence(y, t)
                    acc = F.accuracy(y, F.argmax(t, axis=1))
                else:
                    loss = F.softmax_cross_entropy(y, t)
                    acc = F.accuracy(y, t)

                loss.backward()
                self.optimizer.update()
                train_loss += float(loss.data) * len(t.data)
                train_acc += float(acc.data) * len(t.data)

                elapsed_time = time.time() - self.start_time
                progress = (self.n_batches * (epoch - 1) + i + 1) * 1.0 / (self.n_batches * self.opt.nEpochs)
                eta = elapsed_time / progress - elapsed_time

                line = '* Epoch: {}/{} ({}/{}) | Train: LR {} | Time: {} (ETA: {})'.format(
                    epoch, self.opt.nEpochs, i + 1, self.n_batches,
                    self.optimizer.lr, utils.to_hms(elapsed_time), utils.to_hms(eta))
                sys.stderr.write('\r\033[K' + line)
                sys.stderr.flush()

            self.train_iter.reset()
            train_loss /= len(self.train_iter.dataset)
            train_top1 = 100 * (1 - train_acc / len(self.train_iter.dataset))

            return train_loss, train_top1

    def val_testOnly(self):
        # y_true=[]
        # y_pred=[]

        record = {}
        class_id_to_name = {
            0: "knock", 1: "drawer", 2: "clear throat", 3: "phone", 4: "keys drop",
            5: "speech", 6: "keyboard", 7: "page turn", 8: "cough", 9: "door slam", 10: "laughter"
        }
        self.model.train = False
        with chainer.using_config('train', self.model.train), chainer.no_backprop_mode():#configuration for chainer7.8.1

            val_acc = 0
            for batch in self.val_iter:

                if len(batch[0]) == 3:
                    xs, ts, elevs = zip(*batch)
                    x_array = chainer.dataset.convert.concat_examples(xs)
                    t_array = chainer.dataset.convert.concat_examples(ts)
                    elev_array = chainer.dataset.convert.concat_examples(elevs)

                else:
                    x_array, t_array = chainer.dataset.concat_examples(batch)

                if self.opt.nCrops > 1:
                    # print("Before reshape:", x_array.shape) # x_array shape: (opt.batchSize//nCrops, opt.nCrops, length, number of input channels)
                    if x_array.ndim == 4:
                        x_array = x_array.transpose(0, 1, 3, 2)  #  x_array shape: (opt.batchSize//nCrops, opt.nCrops, number of input channels, length)
                        x_array = x_array.reshape((x_array.shape[0]*self.opt.nCrops, x_array.shape[2], x_array.shape[3]))
                        x_array = x_array[:, :, None, :]
                    else:
                        x_array = x_array.reshape((x_array.shape[0]*self.opt.nCrops, x_array.shape[2]))
                        x_array = x_array[:, None, None, :]

                # print("shape of x_array:", x_array.shape) # shape: ((opt.batchsize//opt.nCrops)*opt.nCrops, number of input channels, 1, length)
                # print("shape of t_array:", t_array.shape) # shape: (opt.batchsize//opt.nCrops, )
                # print("shape of elev_array:", elev_array.shape) # shape: (opt.batchsize//opt.nCrops, )
                # break

                n_batch = x_array.shape[0]// self.opt.nCrops
                x = chainer.Variable(data = cuda.to_gpu(x_array), requires_grad = False)
                t = chainer.Variable(data = cuda.to_gpu(t_array), requires_grad = False)

                y = F.softmax(self.model(x))
                # print("shape of y before reshape:", y.shape) # shape: ((opt.batchsize//opt.nCrops)*opt.nCrops, opt.nClasses)
                y = F.reshape(y, (y.shape[0] // self.opt.nCrops, self.opt.nCrops, y.shape[1]))
                # print("shape of y after reshape:", y.shape) # shape: (opt.batchsize//opt.nCrops, opt.nCrops, opt.nClasses)
                y = F.mean(y, axis=1)
                # print("shape of y after averaging:", y.shape) # shape: (opt.batchsize//opt.nCrops, opt.nClasses)
                # break

                # y_true.extend(t_array)
                # predicted=np.argmax(y.array,axis=1)
                # y_pred.extend(predicted)

                acc = F.accuracy(y, t)
                val_acc += float(acc.data) * len(t.data)

                features = self.model.get_features()
                for b in range(n_batch):
                    sound_class_id = int(t_array[b])
                    class_name = class_id_to_name.get(sound_class_id, str(sound_class_id))
                    angle = int(elev_array[b])

                    # # recording of all crops
                    for layer_name, feat in features.items():
                        # # take out all the crops corresponding to the original waveform
                        crop_feats = feat[b * self.opt.nCrops: (b + 1) * self.opt.nCrops]  # shape: (nCrops, channels, H, W) 或 (nCrops, channels)

                        for crop_idx in range(self.opt.nCrops):
                            # # take out the current single crop feature
                            single_crop_feat = crop_feats[crop_idx]

                            if feat.ndim == 4:
                                activation_per_channel = single_crop_feat.mean(axis=(1, 2))  # average the height(1) and width(temporal length) for each unit
                            else:
                                activation_per_channel = single_crop_feat

                            for neuron_idx, activation in enumerate(activation_per_channel):
                                if layer_name not in record:
                                    record[layer_name] = {}
                                if class_name not in record[layer_name]:
                                    record[layer_name][class_name] = {}
                                if neuron_idx not in record[layer_name][class_name]:
                                    record[layer_name][class_name][neuron_idx] = {}
                                if angle not in record[layer_name][class_name][neuron_idx]:
                                    record[layer_name][class_name][neuron_idx][angle] = []

                                record[layer_name][class_name][neuron_idx][angle].append(float(activation))

                    # # recording of mean value of crops
                    # for layer_name, feat in features.items():
                    #     if feat.ndim == 4:
                    #         crop_feats = feat[b*self.opt.nCrops:(b+1)*self.opt.nCrops]
                    #         activation_per_channel = crop_feats.mean(axis=(0, 2, 3))
                    #     else:
                    #         crop_feats = feat[b * self.opt.nCrops:(b + 1) * self.opt.nCrops]
                    #         activation_per_channel = crop_feats.mean(axis=0)
                    #
                    #     for neuron_idx, activation in enumerate(activation_per_channel):
                    #         if layer_name not in record:
                    #             record[layer_name] = {}
                    #         if class_name not in record[layer_name]:
                    #             record[layer_name][class_name] = {}
                    #         if neuron_idx not in record[layer_name][class_name]:
                    #             record[layer_name][class_name][neuron_idx] = {}
                    #         if angle not in record[layer_name][class_name][neuron_idx]:
                    #             record[layer_name][class_name][neuron_idx][angle] = []
                    #         record[layer_name][class_name][neuron_idx][angle].append(float(activation))

                # Deprecated, previous recording method
                # for layer_name, layer_feat in features.items():
                #     # layer_feat shape: (batch, channels, h, w) or (batch, channels)
                #     if layer_feat.ndim == 4:
                #         avg_feat = layer_feat.mean(axis=(2, 3))  # -> (batch, channels)
                #     elif layer_feat.ndim == 2:
                #         avg_feat = layer_feat  # already (batch, channels)
                #     else:
                #         continue  # skip layers with unexpected shape
                #     n_batch, n_neurons = avg_feat.shape

                    # Initialize storage for this layer
                    # if layer_name not in record:
                    #     record[layer_name] = {}
                    # for i in range(n_batch):
                    #     sound_class_id = int(t_array[i])
                    #     class_name = class_id_to_name.get(sound_class_id, str(sound_class_id))
                    #     elevation = int(elev_array[i])
                    #
                    #     if class_name not in record[layer_name]:
                    #         record[layer_name][class_name] = {}
                    #
                    #     for neuron_idx in range(n_neurons):
                    #         if neuron_idx not in record[layer_name][class_name]:
                    #             record[layer_name][class_name][neuron_idx] = {}
                    #
                    #         if elevation not in record[layer_name][class_name][neuron_idx]:
                    #             record[layer_name][class_name][neuron_idx][elevation] = []
                    #
                    #         record[layer_name][class_name][neuron_idx][elevation].append(
                    #             float(avg_feat[i, neuron_idx])
                    #         )
                # record_feature_batch(features, i, batch_size, record, self.opt.nCrops)
            self.val_iter.reset()
            val_top1 = 100 * (1 - val_acc / len(self.val_iter.dataset))

            return val_top1, record
    def val(self):
        # Deprecated, once used for feature recording
        # feature_sums = {}
        # num_batches = 0
        self.model.train = False
        with chainer.using_config('train', self.model.train), chainer.no_backprop_mode():#configuration for chainer7.8.1

            val_acc = 0
            for batch in self.val_iter:
                x_array, t_array = chainer.dataset.concat_examples(batch)
                if self.opt.nCrops > 1:

                    if x_array.ndim == 4:
                        x_array = x_array.transpose(0, 1, 3, 2)
                        x_array = x_array.reshape((x_array.shape[0]*self.opt.nCrops, x_array.shape[2], x_array.shape[3]))
                        x_array = x_array[:, :, None, :]
                    else:
                        x_array = x_array.reshape((x_array.shape[0]*self.opt.nCrops, x_array.shape[2]))
                        x_array = x_array[:, None, None, :]

                x = chainer.Variable(data = cuda.to_gpu(x_array), requires_grad = False)
                t = chainer.Variable(data = cuda.to_gpu(t_array), requires_grad = False)

                y = F.softmax(self.model(x))
                y = F.reshape(y, (y.shape[0] // self.opt.nCrops, self.opt.nCrops, y.shape[1]))
                y = F.mean(y, axis=1)
                acc = F.accuracy(y, t)
                val_acc += float(acc.data) * len(t.data)


                # Deprecated, calculate the batch's average feature
                # for layer_name, feature in features.items():
                #     feature = cp.asnumpy(feature)  # 转到 CPU
                #     #print(f"Layer: {layer_name}, Feature shape before mean: {feature.shape}")
                #     batch_mean = feature.mean(axis=0)  # 对 batch 维度求平均
                #     if layer_name not in feature_sums:
                #         feature_sums[layer_name] = batch_mean
                #     else:
                #         feature_sums[layer_name] += batch_mean
                # num_batches += 1

            self.val_iter.reset()
            val_top1 = 100 * (1 - val_acc / len(self.val_iter.dataset))
            # Deprecated, calculate the average features
            #avg_features = {layer: feature_sums[layer] / num_batches for layer in feature_sums}

            return val_top1


    def lr_schedule(self, epoch):
        divide_epoch = np.array([self.opt.nEpochs * i for i in self.opt.schedule])
        decay = sum(epoch > divide_epoch)
        if epoch <= self.opt.warmup:
            decay = 1

        return self.opt.LR * np.power(0.1, decay)
