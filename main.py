#!/usr/bin/env python3

#-----------------------------------
#  @nagayosi 2018.2.11
#
#  how to use:    python3 main.py --train --iter 1000 --test
#  if you only train:  python3 main.py --train --iter 1000
#  if you only test:   python3 main.py --test
#-----------------------------------

import chainer
from chainer.dataset import convert
import chainer.links as L
import chainer.functions as F
from chainer.links import Convolution2D as Conv2D, Deconvolution2D as Deconv2D

import glob, argparse, random, os, time
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import _pickle as pickle


## Training data directory path
HOME = os.path.expanduser('~') + '/'
Train_file = 'Dataset/pokemon_train.txt'
pokemon_file = 'Dataset/pokemon.txt'
waza_file = 'Dataset/pokemon_waza.txt'
seikaku_file = 'Dataset/pokemon_seikaku.txt'


## Config containing hyper-parameters
cf = {
    'Minibatch': 10,
    'LearningRate': 0.01,
    'WeightDecay':0.0005,
    'FineTuning': False,
    'SaveModel': 'MyNet.npz'
}

## Network model
class MyNet(chainer.Chain):
    def __init__(self):
        super(MyNet, self).__init__()
        with self.init_scope():
            self.fc1 = L.Linear(None, 1024, nobias=True)
            self.fc2 = L.Linear(None, 1024, nobias=True)
            self.out_doryokuchi = L.Linear(None, 6, nobias=True)
            self.out_seikaku = L.Linear(None, 25, nobias=True)
            self.out_waza = L.Linear(None, 668, nobias=True)

    def acti(self, x):
        return F.tanh(x)
            
    def __call__(self, x):
        fc1 = self.acti(self.fc1(x))
        fc2 = self.acti(self.fc2(fc1))
        out1 = F.sigmoid(self.out_doryokuchi(fc2))
        out2 = self.out_seikaku(fc2)
        out3 = F.sigmoid(self.out_waza(fc2))
        return out1, out2, out3


## Image Load function
def load_data(shuffle=True):

    datas = [x.strip() for x in open(Train_file, 'r').readlines()]
    pokemon = [x.strip() for x in open(pokemon_file, 'r').readlines()]
    waza = [x.strip() for x in open(waza_file, 'r').readlines()]
    seikaku = [x.strip() for x in open(seikaku_file, 'r').readlines()]
    

    for i, line in enumerate(datas):
        item = line.split(',')
        if len(item) < 12:
            continue

        name = np.zeros(len(pokemon), dtype=np.float32)
        name[pokemon.index(item[0])] = 1.
        d = np.array(list(map(int, item[1:7]))) / 4. / 63.
        s = seikaku.index(item[7])
        w = np.zeros(len(waza), dtype=np.float32)
        for j in item[8:]:
            w[waza.index(j)] = 1.
            
        x = np.array((name)).astype(np.float32)
        t = np.hstack((d, s, w)).astype(np.float32)

        if i == 0:
            data1 = x
            data2 = t
        else:
            data1 = np.vstack((data1, x))
            data2 = np.vstack((data2, t))

    if shuffle: 
        inds = np.arange(len(datas))
        random.shuffle(inds)
        data1 = data1[inds]
        data2 = data2[inds]
    
    data = [data1, data2]

    return data


## Fine-tuning function
def copy_model(src, dst):
    assert isinstance(src, chainer.Chain)
    assert isinstance(dst, chainer.Chain)
    for child in src.children():
        if child.name not in dst.__dict__: continue
        dst_child = dst[child.name]
        if type(child) != type(dst_child): continue
        if isinstance(child, chainer.Chain):
            copy_model(child, dst_child)
            if isinstance(child, chainer.Link):
                match = True
                for a, b in zip(child.namedparams(), dst_child.namedparams()):
                    if a[0] != b[0]:
                        match = False
                        break
                    if a[1].data.shape != b[1].data.shape:
                        match = False
                        break
                    if not match:
                        print('Ignore %s because of parameter mismatch' % child.name)
                        continue
                    for a, b in zip(child.namedparams(), dst_child.namedparams()):
                        b[1].data = a[1].data
                        print('Copy %s' % child.name)


def get_batch(data, batch, last):

    ins, gts = data

    data_num = len(ins)
    ind = last + batch

    if ind < data_num:
        in_data = ins[last : ind]
        gt = gts[last : ind]
        last = ind
    else:
        resi = ind - data_num
        in1, gt1 = ins[last:], gts[last:]

        inds = np.arange(len(ins))
        random.shuffle(inds)
        ins = ins[inds]
        gts = gts[inds]
        data = [ins, gts]

        in2, gt2 = ins[:resi], gts[:resi]
        in_data = np.vstack((in1, in2))
        gt = np.vstack((gt1, gt2))
        last = resi

    return in_data, gt, last, data


def parse(data):
    d = data[:, :6].astype(np.float32)
    s = data[:, 6].astype(np.int32)
    w = data[:, 7:].astype(np.float32)
    return d, s, w


## Train function
def main_train(args):

    ## Prepare Images
    train = load_data()
    test = load_data()

    if len(train) < 1 or len(test) < 1:
        raise Exception('train num : {}, test num: {}'.format(len(train), len(test)))
    
    train_count = len(train)
    test_count = len(test)

    print('# train images: {}'.format(train_count))
    print('# test images: {}'.format(test_count))

    
    ## Prepare Network
    model = MyNet()
    if args.gpu_id >=0:
        model.to_gpu()

    if cf['FineTuning']:
        orig = pickle.load(open("../bvlc_alexnet.pkl", "rb"))
        copy_model(orig, model)
        #serializers.load_npz("result/mynet_epoch_100.model", model)

    ## Prepare Optimizer
    optimizer = chainer.optimizers.MomentumSGD(cf['LearningRate'])
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer.WeightDecay(cf['WeightDecay']))


    ## Training start!!
    sum_accuracy_train = 0
    sum_loss_train = 0
    start = time.time()
    
    print('epoch  train_loss  train_accuracy  test_loss  test_accuracy  Elapsed-Time')
    
    last = 0

    for i in range(args.iter):
        i += 1

        x, y, last, train = get_batch(train, cf['Minibatch'], last)
        d, s, w = parse(y)

        train_losses = []
        train_accuracies = []

        x = chainer.Variable(chainer.cuda.to_gpu(x))
        t_d = chainer.Variable(chainer.cuda.to_gpu(d))
        t_s = chainer.Variable(chainer.cuda.to_gpu(s))
        t_w = chainer.Variable(chainer.cuda.to_gpu(w))

        y_d, y_s, y_w = model(x)

        loss_train1 = F.mean_squared_error(y_d, t_d)
        loss_train2 = F.softmax_cross_entropy(y_s, t_s)
        loss_train3 = F.mean_squared_error(y_w, t_w)

        model.cleargrads()
        loss_train1.backward()
        loss_train2.backward()
        loss_train3.backward()
        optimizer.update()

        train_losses.append(chainer.cuda.to_cpu(loss_train1.data))
        #acurracy_train.to_cpu()
        #train_accuracies.append(accuracy_train.data)
        train_accuracies.append(chainer.cuda.to_cpu(loss_train1.data))
        
        #sum_loss_train += float(model.loss.data) * len(t.data)
        #sum_accuracy_train += float(model.accuracy.data) * len(t.data)

        """
        if train_iter.is_new_epoch and train_iter.epoch % 20 == 0:
            #print('epoch: ', train_iter.epoch)
            #print('train mean loss: {:.2f}, accuracy: {:.2f}'.format( sum_loss_train / train_count, sum_accuracy_train / train_count))
            # evaluation

            test_losses = []
            test_accuracies = []

            sum_accuracy_test = 0
            sum_loss_test = 0
            
            #model.predictor.train = False
            for batch in test_iter:
                x_array, t_array = convert.concat_examples(batch,args.gpu_id)
                x = chainer.Variable(x_array)
                t = chainer.Variable(t_array)

                y = model(x)

                loss_test = F.mean_squared_error(y, t)
                #accuracy_test = F.accuracy(y, t)
                
                test_losses.append(chainer.cuda.to_cpu(loss_test.data))
                #accuracy_test.to_cpu()
                #test_accuracies.append(accuracy_test.data)
                test_accuracies.append(chainer.cuda.to_cpu(loss_test.data))

            test_iter.reset()
            #model.predictor.train = True
            #print('test mean  loss: {:.2f}, accuracy: {:.2f}'.format( sum_loss_test / test_count, sum_accuracy_test / test_count))
        
            print('{:>5}  {:^10.4f}  {:^14.4f}  {:^9.4f}  {:^13.4f}  {:^12.2f}'.format(train_iter.epoch, np.mean(train_losses), np.mean(train_accuracies), np.mean(test_losses), np.mean(test_accuracies), time.time()-start))
        """

        print('{:>5} {:^10.4f}'.format(i, np.mean(train_losses)))


    # Save the model and the optimizer
    print('\nsave the model --> {}'.format(cf['SaveModel']) )
    chainer.serializers.save_npz(cf['SaveModel'], model)
    model_name = cf['SaveModel'].split('.')[-2]
    print('save the optimizer --> {}'.format(model_name + '.state'))
    chainer.serializers.save_npz(model_name + '.state', optimizer)
    print()

## Test function    
def main_test(args):
    
    ## Prepare Network
    model = MyNet()
    chainer.serializers.load_npz(cf['SaveModel'], model)

    if args.gpu_id >= 0:
        model.to_gpu()

    ## Test data
    td = ['ガブリアス', 'ボーマンダ']

    pokemon = [x.strip() for x in open(pokemon_file, 'r').readlines()]
    waza = [x.strip() for x in open(waza_file, 'r').readlines()]
    seikaku = [x.strip() for x in open(seikaku_file, 'r').readlines()]


    ## Test start!!
    print('-- test --')
    
    for i in td:
        x = np.zeros(len(pokemon), dtype=np.float32)
        x[pokemon.index(i)] = 1.
        
        # Reshape 1-dimention to [minibatch, data]
        x = x[None, ...]
        
        if args.gpu_id >= 0:
            x = chainer.cuda.to_gpu(x, 0)

        y_d = model(x)[0].data[0]
        y_s = model(x)[1].data[0]
        y_w = model(x)[2].data[0]

        if args.gpu_id >= 0:
            y_d = chainer.cuda.to_cpu(y_d)
            y_s = chainer.cuda.to_cpu(y_s)
            y_w = chainer.cuda.to_cpu(y_w)
        
        print()
        print(i)

        ## Doryokuchi
        d_sum = y_d.sum()
        d = np.round(y_d /d_sum * 126.).astype(np.int) * 4

        print('  H |  A |  B |  C |  D |  S |')
        print('{:>4d}|{:>4d}|{:>4d}|{:>4d}|{:>4d}|{:>4d}|'.format(d[0], d[1], d[2], d[3], d[4], d[5]))

        ## Seikaku
        print('seikaku: {}'.format(seikaku[y_s.argmax()]))

        ## Waza
        for j, w in enumerate(y_w.argsort()[::-1].astype(int)[:4]):
            j += 1
            print('waza{} : {}'.format(j, waza[w]))
        #print(waza[w] for w in y_w[0].argsort()[::-1].astype(int)[:4])



def print_config(args):
    print('-- config parameters --')
    print('GPU ID : {}'.format(args.gpu_id))
    print('Train file : {}'.format(Train_file))
    print('Train pokemon file : {}'.format(pokemon_file))
    print('Train seikaku file : {}'.format(seikaku_file))
    print('Train waza file : {}'.format(waza_file))
    
    for k, v in cf.items():
        print('{} : {}'.format(k, v))
    print('----\n')



def parse_args():
    parser = argparse.ArgumentParser(description='Faster R-CNN demo')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]',default=0, type=int)
    parser.add_argument('--cpu', dest='cpu_mode',help='Use CPU (overrides --gpu)',action='store_true')
    parser.add_argument('--train', dest='train', help='train', action='store_true')
    parser.add_argument('--test', dest='test', help='test', action='store_true')
    parser.add_argument('--iter', dest='iter', help='iteration', default=100, type=int)
    args = parser.parse_args()
    return args



if __name__ == '__main__':

    args = parse_args()
    print_config(args)

    if args.cpu_mode:
        pass
    else:
        chainer.cuda.get_device(args.gpu_id).use()
    
    if args.train:
        main_train(args)
    if args.test:
        main_test(args)



