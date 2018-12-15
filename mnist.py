import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

import numpy as np
import itertools

import matplotlib.pyplot as plt
import math

def epochs_to_batches(ds, epochs, bs):
    return epochs * (len(ds) // bs)

def by_epoch(ds, epochs=1, bs=1, shuffle=True):
    for epoch in range(epochs):
        batcher = torch.utils.data.DataLoader(ds, batch_size=bs, shuffle=shuffle, drop_last=True)
        for data, labels in batcher:
            yield (data, labels)

def by_batch(ds, batches=1, bs=1, shuffle=True):
    n = 1
    while True:
        for n, (data, labels) in enumerate(by_epoch(ds, 1, bs, shuffle), n) :
            yield (data, labels)
            if n == batches:
                return

class Callback():
    def __init__(self, learner):
        self.learner = learner

    def on_train_begin(self):
        pass
    def on_train_step(self, step, loss, rate, mom, xs, ys):
        pass
    def on_train_end(self):
        pass
    
class StandardCallback(Callback):
    def __init__(self, learner):
        self.learner = learner

    def on_train_begin(self):
        print("training begins")
        self.losses = []
        self.rates =  []

    def accuracy_report(self, dataset, tag, bs):
        acc, lss = accuracy_t(self.learner.net, self.learner.loss, dataset)
        print(f"{tag}: loss = {lss:.3g}, accuracy = {percent(acc)}")

    def report(self):
        self.accuracy_report(self.learner.validation_set,
                             "  validation", bs=50)
        
    def on_train_step(self, step, loss, rate, mom, xs, ys, report_every):
        self.losses.append(loss)
        self.rates.append(rate)
        step += 1

        if (step == 1) or (step % report_every == 0):
            acc = accuracy(self.learner.classify(xs.numpy()), ys.numpy())
            print(f"batch {step}: loss = {loss:.3g}, lr = {rate:.3g}, p = {mom:.3g}, accuracy = {percent(acc)}")

        if (step % (10 * report_every) == 0):
            self.report()

    def on_train_end(self):
        print("training ends")
        self.report()

class AugmentedCallback(StandardCallback):
    def __init__(self, learner, every=40):
        self.learner = learner
        super().__init__(learner)
        self.every = every

    def on_train_begin(self):
        super().on_train_begin()
        self.vlosses = []
        self.acc = []

    def on_train_step(self, step, loss, rate, mom, xs, ys, report_every):
        super().on_train_step(step, loss, rate, mom, xs, ys, report_every)
        # expensive, use for collecting when needed
        if (step % self.every) == 0:
            acc, l = accuracy_t(self.learner.net, self.learner.loss, self.learner.validation_set)
            self.vlosses.append(l)
            self.acc.append(acc)

    def on_train_end(self):
        def plotit(losses, xoffset):
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.plot(self.every * (xoffset + np.arange(len(losses))), losses)
            ax.grid(True)
            ax.set_ylabel("validation loss")
            ax.set_xlabel("step")
            plt.show()

        half = len(self.vlosses) // 2
        plotit(self.vlosses[:half], 0)
        plotit(self.vlosses[half:], half)
        super().on_train_end()

class Learner():
    def __init__(self, model, train_set, validation_set):
        self.train_set = train_set
        self.validation_set = validation_set
        self.stop_requested = False
        self.net = model.cuda()
        self.loss = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.net.parameters(), lr=6.25e-4, eps=1e-8, betas=(0.90, 0.99), weight_decay=0)

    # Optional batch size bs chops up the calculation
    # into feasible sizes for large datasets on large models.
    #   mode==plain: each prediction is a label
    #   mode==softmax: each prediction is a probability distribution over the labels
    #   mode==raw: each prediction is the raw output from the net, a pre-softmax vector
    def classify(self, x, bs=None, mode="plain"):
        if bs is None: bs = len(x)
        batches = []
        frm = 0
        self.net.eval()
        while True:
            to = min(frm + bs, len(x))
            xt = torch.tensor(x[frm:to]).to("cuda")
            y = self.net(xt)
            if mode=="softmax":
                y = F.softmax(y, 1)
            batches.append(y.cpu().detach().numpy())
            frm = to
            if to == len(x): break

        y = np.concatenate(batches)
        if mode=="plain":
            return np.argmax(y, 1)
        else:
            return y

    def request_stop(self):
        self.stop_requested = True
        
    def train(self, epochs=1, batches=None, bs=64, lr=0.03, p=0.90, report_every=100, callback=None, **kwargs):

        self.stop_requested = False;
        if callback is None:
            callback = StandardCallback(self)

        if epochs is not None:
            print(f"epochs: {epochs}, batch_size: {bs}, batches: {epochs_to_batches(self.train_set, epochs, bs)}")
            batcher = by_epoch(self.train_set, epochs, bs, shuffle=True)
        else:
            print(f"batch_size: {bs}, batches: {batches}")
            batcher = by_batch(self.train_set, batches, bs, shuffle=True)

        callback.on_train_begin()

        for i, (xs, ys) in enumerate(batcher):
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

            batch = xs.to("cuda")
            labels = ys.to("cuda")

            self.net.train()

            pred = self.net(batch)
            ce_loss = self.loss(pred, labels)
            self.optimizer.zero_grad()
            ce_loss.backward()
            self.optimizer.step()

            callback.on_train_step(i, ce_loss.data.item(), learning_rate, momentum, xs, ys, report_every)

        callback.on_train_end()
        return callback


class Evaluator():
    def __init__(self, model, device="cuda"):
        self.device = device
        self.model = model.to(device)

    def eval(self, x):
        return self.model(x.to(self.device))

    def __call__(self, x):
        return self.softmax(x).max(1, keepdim=False)[1]

    def logits(self, x):
        return self.eval(x)

    def softmax(self, x):
        return F.softmax(self.eval(x), 1)

def accuracy_t(model, lossftn, ds, bs=100):
    nn = Evaluator(model)
    batcher = by_epoch(ds, epochs=1, bs=bs)
    correct = 0
    tloss = 0
    for n, (batch, labels) in enumerate(batcher, 1):
        batch = batch.cuda()
        labels = labels.cuda()
        pred = nn(batch)
        correct += pred.eq(labels.view_as(pred)).sum().item()
        logits = nn.logits(batch)
        tloss += lossftn(logits, labels).item()

    accuracy = correct / (n * bs)
    loss = tloss / n
    return accuracy, loss

bn_params = {'eps' : 1e-5, 'momentum' : 0.1} # pytorch defaults
#bn_params = {'eps' : 0.001, 'momentum' : 0.99}# tensorflow defaults

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

def mnist_classifier():
    return nn.Sequential(
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
               # Softmax provided during training.
           )

def exponential_decay(v, half_life):
    return lambda n: v * np.power(1/2, n/half_life)

def exponential_rise(v, decades, steps):
    x = np.power(10, decades/steps)
    return lambda n: v * np.power(x, n)

def linear(steps, left, right):
    m = (right-left)/steps
    return lambda n: m*n + left

def triangular(steps, left, middle, right=None):
    if right is None: right=left
    f = linear(steps, left, middle)
    g = linear(steps, middle, right)
    return lambda n: f(n) if n<=steps else g(n-steps)

def one_cycle(learner, epochs, lrmin, lrmax, bs, pmax=0.95, pmin=0.85, 
              lrmin2=None, pmax2=None, callback=None, **kwargs):
    steps = epochs_to_batches(learner.train_set, epochs, bs)
    if lrmin2 is None: lrmin2=lrmin
    if pmax2 is None: pmax2 = pmax
    return learner.train(epochs=2*epochs,
                         bs=bs,
                         lr=triangular(steps, left=lrmin, middle=lrmax, right=lrmin2),
                         p=triangular(steps, left=pmax, middle=pmin, right=pmax2),
                         callback=callback,
                         report_every=steps//10)

def percent(n):
    return "%.2f%%" % (100 * n)

def show_image(v, title=None):
    if type(v) is torch.Tensor:
        v = v.numpy()
    v = v.reshape([28,28])
    plt.figure()
    plt.title(title)
    plt.xticks([])
    plt.yticks([])
    img = plt.imshow(v, cmap="Greys", )

def plot_images(batch, title=None):
    if type(batch) is torch.Tensor:
        batch = batch.cpu().detach().numpy()
    n = batch.shape[-1]
    batch = batch.reshape(-1, n, n)
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

def accuracy(predictions, actual):
    errors = np.where(predictions != actual)[0]
    return 1.0 - len(errors) / len(actual)

def lr_find(learner, bs, batches=500, start=1e-6, decades=7, steps=500, **kwargs):
    class LR_Callback(Callback):
        def __init__(self, learner):
            super().__init__(learner)
            self.losses = []
            self.rates = []
            self.best = math.inf

        def on_train_step(self, step, loss, rate, mom, xs, ys, report_every):
            if (loss > 4 * self.best) and (loss > 2) and (step > 50):
                self.learner.request_stop()
            self.best = min(self.best, loss)
            self.losses.append(loss)
            self.rates.append(float(rate))

    recorder = LR_Callback(learner)
    lr = exponential_rise(start, decades, steps)
    learner.train(epochs=None, batches=steps, bs=bs, lr=lr, callback=recorder, **kwargs)

    plt.semilogx(recorder.rates, recorder.losses)
    plt.xlabel("Learning Rate")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.show()
    return recorder

def one_hot(d, cols=10):
    b = np.zeros([d.shape[0], cols])
    b[np.arange(b.shape[0]), d] = 1
    return b

def vote(classifiers, n_classes, data):
    r = np.zeros([data.shape[0], n_classes])
    for cl in classifiers:
        #r += cl.classify(data, bs=5000, mode="softmax")
        r += one_hot(cl.classify(data, bs=500))
    return np.argmax(r, axis=1)

def voting_classifier(classifiers, n_classes):
    return lambda data : vote(classifiers, n_classes, data)

def committee(classifiers, dataset):
    vclass = voting_classifier(classifiers, 10)
    acc = accuracy(vclass(dataset.data), dataset.labels)
    print(f"committee of {len(classifiers)} accuracy: {100 * acc}")
    return acc

def create_mnist_datasets():
    def img_normalize(t):
        c, x, y = t.shape
        t = t.view(c, -1)
        t = t - t.mean(dim=1, keepdim=True)
        t = t / t.std(dim=1, keepdim=True, unbiased=False)
        return t.view(c, x, y)

    xform = transforms.Compose([
                          transforms.ToTensor(),
                          transforms.Lambda(img_normalize)
                       ])

    training_set = datasets.MNIST('./data', train=True, download=True, transform=xform)
    test_set = datasets.MNIST('./data', train=False, download=True, transform=xform)
    return (training_set, test_set)

def main():
    tset, vset = create_mnist_datasets()

    classifiers = [Learner(mnist_classifier(), tset, vset) for i in range(35)]
    params = {'epochs': 4, 'bs': 100,
              'lrmin': 1e-4, 'lrmax': 1e-3,
              'pmax' : 0.95, 'pmin' : 0.70, 'pmax2' : 0.70}

    for classifier in classifiers:
        one_cycle(classifier, **params)

    committee(classifiers, vset)

if __name__ == "__main__":
    main()
    
