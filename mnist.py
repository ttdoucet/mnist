import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np
import input_data

import matplotlib.pyplot as plt
import math

class MNIST_DataSet():
    image_size = 28
    n_labels = 10

    @staticmethod
    def normalize(batch):
        mean = np.mean(batch, axis=1).reshape(-1,1)
        std = np.std(batch, axis=1).reshape(-1,1)
        return (batch - mean) / std

    def __init__(self, path, normalize=True):
        self.mnist = input_data.read_data_sets(path, one_hot=False)

        if normalize is True:
            # This is kind of unholy but I want to do it only once.
            self.mnist.train._images = self.normalize(self.mnist.train._images)
            self.mnist.validation._images = self.normalize(self.mnist.validation._images)
            self.mnist.test._images = self.normalize(self.mnist.test._images)

    def next_batch(self, n):
        batch, labels = self.mnist.train.next_batch(n)
        b = batch.reshape([n, -1, 28, 28])
        return (b, labels.astype("int64"))

    @property
    def validation_images(self):
        imgs = self.mnist.validation.images
        return imgs.reshape(imgs.shape[0], -1, 28, 28)

    @property
    def validation_labels(self):
        return self.mnist.validation.labels.astype("int64")

    @property
    def test_images(self):
        imgs = self.mnist.test.images
        return imgs.reshape(imgs.shape[0], -1, 28, 28)

    @property
    def test_labels(self):
        return self.mnist.test.labels.astype("int64")
    
    @property
    def epoch_size(self):
        return len(self.mnist.train.labels)
    def epochs_to_batches(self, epochs, bs):
        return self.divide_round_up(epochs * self.epoch_size, bs)
    @staticmethod
    def divide_round_up(n, d):
        return (n + (d-1))//d

class Callback():
    def on_train_begin(self):
        pass
    def on_train_step(self, step, loss, rate, mom, xs, ys):
        pass
    def on_train_end(self):
        pass

class LR_Callback(Callback):
    def __init__(self, learner):
        self.learner = learner
        self.losses = []
        self.rates = []
        self.best = math.inf

    def on_train_begin(self):
        self.losses = []
        self.rates = []

    def on_train_step(self, step, loss, rate, mom, xs, ys, report_every):
        if (loss > 4 * self.best) and (loss > 2) and (step > 50):
            self.learner.request_stop()

        self.best = min(self.best, loss)
        self.losses.append(loss)
        self.rates.append(float(rate))
    
class StandardCallback(Callback):
    def __init__(self, learner):
        self.learner = learner

    def on_train_begin(self):
        print("training begins")
        self.losses = []
        self.rates =  []

    def report(self):
        accuracy_report(self.learner, self.learner.dataset.validation_images,
                        self.learner.dataset.validation_labels, "  validation", bs=64)
        accuracy_report(self.learner, self.learner.dataset.test_images,
                        self.learner.dataset.test_labels, "        test", bs=64)

        
    def on_train_step(self, step, loss, rate, mom, xs, ys, report_every):
        self.losses.append(loss)
        self.rates.append(rate)
        step += 1

        if (step == 1) or (step % report_every == 0):
            acc = accuracy(self.learner.classify(xs), ys)
            print(f"batch {step}: loss = {loss:.3g}, lr = {rate:.3g}, p = {mom:.3g}, accuracy = {percent(acc)}")

        if (step % (10 * report_every) == 0):
            self.report()

    def on_train_end(self):
        print("training ends")
        self.report()

class AugmentedStandardCallback(StandardCallback):
    def __init__(self, learner):
        self.learner = learner
        super().__init__(learner)

    def on_train_begin(self):
        super().on_train_begin()
        self.vlosses = []
        self.acc = []

    def on_train_step(self, step, loss, rate, mom, xs, ys, report_every):
        super().on_train_step(step, loss, rate, mom, xs, ys, report_every)
        # expensive, use for collecting when needed
        l, a = self.learner.get_loss(self.learner.dataset.validation)
        self.vlosses.append(l)
        self.acc.append(a)
        
    def on_train_end(self):
        super().on_train_end()


class Learner():
    def __init__(self, dataset, name=None, init=False):
        self.dataset = dataset
        self.stop_requested = False
        self.net = self.model().to("cuda")
        self.loss = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.net.parameters(), lr=6.25e-4, eps=1e-8, betas=(0.95, 0.99), weight_decay=0)

    def model(self):
        raise NotImplementedError

    # Optional batch size bs chops up the calculation
    # into feasible sizes for large datasets on large models.
    def classify(self, x, bs=None, softmax=False):
        if bs is None: bs = len(x)
        batches = []
        frm = 0
        #x = torch.tensor(x).to("cuda")
        x = torch.tensor(x)
        while True:
            to = min(frm + bs, len(x))
            # y = F.softmax( self.net.eval()(x[frm:to]).to("cuda"), 1 )
            y = F.softmax( self.net.eval()(x[frm:to].to("cuda")), 1 )
            batches.append(y.cpu().detach().numpy())
            frm = to
            if to == len(x): break
        y = np.concatenate(batches)

        if softmax is True:
            return y
        else:
            return np.argmax(y, 1)


    def request_stop(self):
        self.stop_requested = True
        
    def train(self, epochs=None, batches=1, bs=1, lr=0.03, p=0.90, report_every=100, callback=None, save=True, **kwargs):

        self.stop_requested = False;
        if callback is None:
            callback = StandardCallback(self)

        if epochs is not None:
            batches = self.dataset.epochs_to_batches(epochs, bs)
        print("epochs: {}, batches: {}, batch_size: {}".format(epochs, batches, bs))

        callback.on_train_begin()
        for i in range(batches):
            if self.stop_requested is True:
                print(f"train: stop requested at step {i}")
                break

            learning_rate=lr(i) if callable(lr) else lr
            momentum = p(i) if callable(p) else p

            for param_group in self.optimizer.param_groups:
                param_group['lr'] = learning_rate
                # specific to Adam
                betas = param_group['betas']
                param_group['betas'] = (momentum, betas[1])
                #print("param lr:", param_group['lr'])
                #print("param betas:", param_group['betas'])

            xs, ys = self.dataset.next_batch(bs)
            batch = torch.from_numpy(xs).to("cuda")
            labels = torch.from_numpy(ys).to("cuda")

            pred = self.net(batch)
            ce_loss = self.loss(pred, labels)
            self.optimizer.zero_grad()
            ce_loss.backward()
            self.optimizer.step()

            # dict = {self.x: xs, self.y: ys,
            #         self.keep_prob: self.training_keep_prob,
            #         self.is_training: True,
            #         self.rate : learning_rate,
            #         self.mom : momentum}
            # _, loss, rate, mom = sess.run([self.optimizer, self.loss, self.rate, self.mom], feed_dict=dict)

            callback.on_train_step(i, ce_loss.data.item(), learning_rate, momentum, xs, ys, report_every)

        callback.on_train_end()
        return callback
   # end train()
# end class Learner()


# pytorch defaults
# bn_params = {'eps' : 1e-5, 'momentum' : 0.1}

# tensorflow defaults
bn_params = {'eps' : 0.001, 'momentum' : 0.99}

class Residual(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.bn = nn.BatchNorm2d(d, **bn_params)
        self.conv3x3 = nn.Conv2d(in_channels=d, out_channels=d, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.bn(x)
        return x + F.relu(self.conv3x3(x))

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size()[0], -1)

class LearnerV2_a(Learner):
    def __init__(self, dataset, name=None, init=False):
        super().__init__(dataset, name, init)

    def model(self):
        v = nn.Sequential(
                nn.Conv2d(in_channels=1, out_channels=128, kernel_size=5, padding=2),
                nn.ReLU(),

                Residual(128),
                nn.MaxPool2d(2),
                Residual(128),

                nn.BatchNorm2d(128, **bn_params),
                nn.Conv2d(128, 256, 3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                Residual(256),

                nn.BatchNorm2d(256, **bn_params),
                nn.Conv2d(256, 512, 3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2, ceil_mode=True),
                Residual(512),

                nn.BatchNorm2d(512, **bn_params),
                nn.AvgPool2d(kernel_size=4),
                Flatten(),
                nn.Linear(512,10),
                # nn.Softmax(1)
        )
        return v
    # end of model()
# end class LearnerV2_A()

def exponential_decay(v, half_life):
    return lambda n: v * np.power(1/2, n/half_life)

def exponential_rise(v, decades, steps):
    x = np.power(10, decades/steps)
    return lambda n: v * np.power(x, n)

def linear(steps, left, right):
    m = (right-left)/steps
    return lambda n: m*n + left

def triangular(steps, left, middle):
    f = linear(steps, left, middle)
    return lambda n: f(n) if n<=steps else f(2*steps-n)

def one_cycle(learner, epochs, lrmin, lrmax, bs, callback=None, **kwargs):
    steps = learner.dataset.epochs_to_batches(epochs, bs)
    return learner.train(epochs=2*epochs,
                         bs=bs,
                         lr=triangular(steps, left=lrmin, middle=lrmax),
                         p=triangular(steps, left=0.95, middle=0.85),
                         callback=callback,
                         report_every=steps//10)

def percent(n):
    return "%.2f%%" % (100 * n)

def show_image(v, title=None):
    v = v.numpy()
    vv = v.reshape([28,28])
    plt.figure()
    plt.title(title)
    plt.xticks([])
    plt.yticks([])
    img = plt.imshow(vv, cmap="Greys", )

def plot_images(batch, title=None):
    batch = batch.numpy()
    batch = batch.reshape(-1, 28, 28)
    n = batch.shape[0]
    s = int(math.sqrt(n) + 0.5)
    if (s*s < n):
        s += 1

    fig = plt.figure(figsize=(10,10))
    fig.canvas.set_window_title(title)
    for i in range(n):
        plt.subplot(s, s, i+1)
        plt.axis('off')
        plt.imshow(batch[i,:,:], cmap="Greys")

def error_rate(predictions, actual):
    errors = np.where(predictions != actual)[0]
    return len(errors) / len(actual)

def accuracy(predictions, actual):
    return 1.0 - error_rate(predictions, actual)

def accuracy_report(learner, images, labels, tag, bs=None):
    acc = accuracy(learner.classify(images, bs=bs), labels)
    print(f"{tag}: accuracy = {percent(acc)}")

def lr_find(learner, bs, batches=500, decades=6, start=0.0001, steps=500, **kwargs):
    recorder = LR_Callback(learner)
    lr = exponential_rise(start, decades, steps)
    learner.train(epochs=None, batches=steps, bs=bs, lr=lr, callback=recorder, save=False, **kwargs)
    #learner.restore()
    plt.semilogx(recorder.rates, recorder.losses)
    plt.xlabel("Learning Rate")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.show()
    return recorder

def main():
    data = MNIST_DataSet("MNIST_data/")
    # nn = LearnerV2_a(data)
    # lr_find(nn, bs=64, start=1e-8, decades=9)

    nn = LearnerV2_a(data)
    params = {'epochs': 4, 'bs': 64, 'lrmin': 1.875e-05, 'lrmax': 6.25e-4}
    cb = one_cycle(nn, **params)
    
    stepsize = data.epochs_to_batches(params['epochs'], params['bs'])
    plt.plot(cb.losses[:stepsize])
    plt.grid(True)
    plt.show()
    plt.plot(cb.losses[stepsize:])
    plt.grid(True)
    plt.show()

    print("That's all Folks!")


if __name__ == "__main__":
    main()
    
