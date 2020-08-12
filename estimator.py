from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import tensorflow.contrib.eager as tfe
from utils import loadData

import os
import argparse

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

config = tf.ConfigProto()
tf.enable_eager_execution(config=config, device_policy=tfe.DEVICE_PLACEMENT_SILENT)

tf.set_random_seed(2018)
np.random.seed(2018)
batch_size = 128
n_cat = 5


class KidneyModel(tf.keras.Model):
    def __init__(self, n_cat):
        super(KidneyModel, self).__init__()
        self.n_cat = n_cat
        self.dense1 = tf.keras.layers.Dense(units=64,
                                            activation=tf.nn.leaky_relu)
        self.dense2 = tf.keras.layers.Dense(units=128,
                                            activation=tf.nn.leaky_relu)
        self.dense3 = tf.keras.layers.Dense(units=128,
                                            activation=tf.nn.leaky_relu)
        self.dense4 = tf.keras.layers.Dense(units=128,
                                            activation=tf.nn.leaky_relu)
        self.dense_mean = tf.keras.layers.Dense(units=64,
                                                activation=tf.nn.leaky_relu)
        self.dense_var = tf.keras.layers.Dense(units=64,
                                               activation=tf.nn.leaky_relu)
        self.dense_cat = tf.keras.layers.Dense(units=64,
                                               activation=tf.nn.leaky_relu)
        self.mean = tf.keras.layers.Dense(units=1)
        self.var = tf.keras.layers.Dense(units=1,
                                         activation=tf.exp)
        self.logits = tf.keras.layers.Dense(units=n_cat-1,
                                            activation=tf.sigmoid)
        self.dense_igfr = tf.keras.layers.Dense(units=64,
                                                activation=tf.nn.leaky_relu)
        self.igfr = tf.keras.layers.Dense(units=1)

    def call(self, input):
        x = self.dense1(input)
        x = self.dense2(x)
        x = self.dense3(x)
        var = self.var(self.dense_var(x))
        mean = self.mean(self.dense_mean(x))
        # xx = self.dense4(x)
        logits = self.logits(x)
        igfr = self.igfr(self.dense_igfr(x))
        return mean, var, logits, igfr


def loss(mean, var, logits, igfr, labels, log_igfr, enlarge, weight_div, weight_l2):
    gaussian_loss = tf.reduce_mean(
        tf.divide(tf.square(enlarge*tf.squeeze(mean)-enlarge*log_igfr), var) +
        tf.log(var))
    cross_entropy_loss = tf.nn.softmax_cross_entropy_with_logits_v2(
        labels=labels,
        logits=logits)
    cross_entropy_loss = tf.reduce_mean(cross_entropy_loss)

    div_loss = weight_div * tf.reduce_mean(
        tf.square(tf.div(tf.squeeze(igfr), tf.exp(log_igfr)) - 1))

    l2_loss = weight_l2 * tf.reduce_mean(
        tf.square(tf.exp(tf.squeeze(mean)) - tf.exp(log_igfr)))
    final_loss = (gaussian_loss + cross_entropy_loss + div_loss + l2_loss) / 4
    #  final_loss = (gaussian_loss + cross_entropy_loss) / 2
    return final_loss  # , gaussian_loss, cross_entropy_loss


def accuracy(logits, labels):
    return tf.reduce_sum(
        tf.cast(
            tf.equal(
                tf.argmax(logits, axis=1,
                          output_type=tf.int32),
                tf.argmax(labels, axis=1,
                          output_type=tf.int32)),
            dtype=tf.float32)) / float(labels.shape[0].value)


def test(model, dataset):
    acc = tfe.metrics.Accuracy('accuracy')
    for (batch, (x, log_igfr, labels)) in enumerate(tfe.Iterator(dataset)):
        print(x)
        print(x.shape)
        _, _, logits, _ = model(x)
        acc(tf.argmax(logits, axis=1, output_type=tf.int32),
            tf.argmax(labels, axis=1, output_type=tf.int32))
    res = 100*acc.result()
    print('Testing acc {}'.format(res))
    return res.cpu().numpy()


def test2(model, dataset):
    n_total, n_correct = 0, 0
    each_total = np.asarray([0.] * 5)
    each_correct = np.asarray([0.] * 5)
    boundaries = [15, 30, 45, 60, 600]
    for (batch, (x, log_igfr, labels)) in enumerate(tfe.Iterator(dataset)):
        n_total += labels.shape[0].value
        mean, var, _, _ = model(x)
        for i in range(labels.shape[0]):
            pred = tf.exp(mean[i]).numpy()
            target = tf.exp(log_igfr[i]).numpy()
            ratio = abs(pred - target) / float(target) * 100
            idx = 0
            for j in range(len(boundaries)):
                idx = j
                if target < boundaries[j]:
                    break
            each_total[idx] += 1

            if ratio <= 30:
                n_correct += 1
                each_correct[idx] += 1
    res = float(n_correct)/n_total * 100
    print('Real Correct acc is {}'.format(res))
    reses = each_correct / each_total * 100
    return res, reses


# use igfr branch
def test3(model, dataset):
    n_total, n_correct = 0, 0
    each_total = np.asarray([0.] * 5)
    each_correct = np.asarray([0.] * 5)
    boundaries = [15, 30, 45, 60, 600]
    for (batch, (x, log_igfr, labels)) in enumerate(tfe.Iterator(dataset)):
        n_total += labels.shape[0].value
        _, _, _, igfr = model(x)
        for i in range(labels.shape[0]):
            pred = igfr[i].numpy()
            target = tf.exp(log_igfr[i]).numpy()
            ratio = abs(pred - target) / float(target) * 100
            idx = 0
            for j in range(len(boundaries)):
                idx = j
                if target < boundaries[j]:
                    break
            each_total[idx] += 1

            if ratio <= 30:
                n_correct += 1
                each_correct[idx] += 1
    res = float(n_correct)/n_total * 100
    print('iGFR Correct acc is {}'.format(res))
    reses = each_correct / each_total * 100
    return res, reses


def test23(model, dataset):
    n_total, n_correct = 0, 0
    n_correct2 = 0
    each_total = np.asarray([0.] * 5)
    each_correct = np.asarray([0.] * 5)
    zscores = []
    rmsds = []

    each_total2 = np.asarray([0.] * 5)
    each_correct2 = np.asarray([0.] * 5)
    boundaries = [15, 30, 45, 60, 600]
    for (batch, (x, log_igfr, labels)) in enumerate(tfe.Iterator(dataset)):
        n_total += labels.shape[0].value
        mean, var, _, igfr = model(x)
        mean = tf.squeeze(mean)
        var = tf.squeeze(var)
        zscore = tf.div(mean - log_igfr, tf.sqrt(var))
        rmsd = tf.square(mean - log_igfr)


        for i in range(labels.shape[0]):
            zscores.append(zscore.numpy()[i])
            rmsds.append(rmsd.numpy()[i])
        for i in range(labels.shape[0]):
            pred = tf.exp(mean[i]).numpy()
            pred2 = igfr[i].numpy()
            target = tf.exp(log_igfr[i]).numpy()
            ratio = abs(pred - target) / float(target) * 100
            ratio2 = abs(pred2 - target) / float(target) * 100
            idx = 0
            for j in range(len(boundaries)):
                idx = j
                if target < boundaries[j]:
                    break
            each_total[idx] += 1

            if ratio <= 30:
                n_correct += 1
                each_correct[idx] += 1
            if ratio2 <= 30:
                n_correct2 += 1
                each_correct2[idx] += 1

    RMSD = np.sqrt(np.mean(np.asarray(rmsds)))
    print('RMSD is ')
    print(RMSD)

    res = float(n_correct)/n_total * 100
    res2 = float(n_correct2)/n_total * 100
    print('Real Correct acc is {}'.format(res))
    reses = each_correct / each_total * 100
    reses2 = each_correct2 / each_total2 * 100
    return res, reses, res2, reses2


def main(args):
    xr, log_igfr_r, labels_r = loadData('NEW_GFR_TRAIN')
    xe, log_igfr_e, labels_e = loadData('NEW_GFR_TEST')

    train_ds = tf.data.Dataset.from_tensor_slices((xr, log_igfr_r, labels_r))
    test_ds = tf.data.Dataset.from_tensor_slices((xe, log_igfr_e, labels_e))

    train_ds = train_ds.shuffle(xr.shape[0]).batch(batch_size)
    # test_ds = test_ds.batch(batch_size)
    test_ds = test_ds.batch(1)

    model = KidneyModel(n_cat)
    init_lr, momentum = args.learning_rate, 0.9
    lr = tfe.Variable(init_lr, name="learning_rate")
    optimizer = tf.train.AdamOptimizer(lr)

    with tf.device('/cpu:0'):
        lr = tfe.Variable(init_lr, name="learning_rate")
        optimizer = tf.train.AdamOptimizer(lr)
        for epoch in range(args.epochs):
            print('epoch', epoch)
            train_acc = tfe.metrics.Accuracy('train_accuracy')
            total_loss, total_batch = 0.0, 0.0
            for (batch, (x, log_igfr, labels)) in enumerate(tfe.Iterator(train_ds)):
                with tf.GradientTape() as tape:
                    mean, var, logits, igfr = model(x)
                    loss_value = loss(
                        mean, var, logits, igfr,
                        labels, log_igfr, args.enlarge, args.w_div, args.w_l2)
                total_loss += loss_value.cpu().numpy()
                total_batch += 1
                train_acc(tf.argmax(logits, axis=1, output_type=tf.int32),
                          tf.argmax(labels, axis=1, output_type=tf.int32))
                grads = tape.gradient(loss_value, model.variables)
                optimizer.apply_gradients(zip(grads, model.variables),
                                          global_step=tf.train.get_or_create_global_step())
            print('Learning Rate', lr.numpy())
            if (epoch + 1) % 50 == 0:
                lr.assign(lr.numpy()/2)

            print('Training acc {}'.format(100*train_acc.result()))
            print('train_acc', 100*train_acc.result().cpu().numpy())
            test_acc = test(model, test_ds)
            test2_acc, reses, test3_acc, reses3 = test23(model, test_ds)
            print('test_acc1', test_acc)
            print('avg_loss ', total_loss / total_batch)
            print('test_acc2', test2_acc)
            print('test_acc3', test3_acc)
            for i in range(reses.shape[0]):
                print('Cate %d ' % i, reses[i])
    checkpoint_dir = './saved_models/'
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    root = tfe.Checkpoint(optimizer=optimizer,
                          model=model,
                          optimizer_step=tf.train.get_or_create_global_step())

    root.save(file_prefix=checkpoint_dir)
    # test_acc = test(model, test_ds)
    # test2_acc, reses, test3_acc, reses3 = test23(model, test_ds)
    # print('test_acc2: {}, test_acc3: {}'.format(test2_acc, test3_acc))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Triple loss')
    parser.add_argument('-e', '--enlarge', type=float, default=25,
                        help='increse the range of log(iGFR)')
    parser.add_argument('-w', '--w_div', type=float, default=5,
                        help='weigh of div component')
    parser.add_argument('--w_l2', type=float, default=0.0,
                        help='weigh of l2 component')
    parser.add_argument('-l', '--learning_rate', type=float, default=0.0005,
                        help='learning rate')
    parser.add_argument('-b', '--batch_size', type=float, default=256,
                        help='batch size')
    parser.add_argument('--epochs', type=int, default=200,
                        help='epochs')
    parser.add_argument('-o', '--optimizer', type=str,
                        default='Adam', help='optimizer')
    args = parser.parse_args()
    enlarge = args.enlarge
    main(args)
