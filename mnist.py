# Written by Todd Doucet.

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
import math
import os
import Augmentor
from  functools import partial
from tqdm.auto import tqdm


def epochs_to_batches(ds, epochs, bs):
    return epochs * (len(ds) // bs)

def Batcher(ds, epochs=None, batches=None, bs=1, shuffle=True):
    "Iterator interface to dataset."

    def by_epoch(ds, epochs=1, bs=1, shuffle=True):
        for epoch in range(epochs):
            batcher = torch.utils.data.DataLoader(ds, batch_size=bs, shuffle=shuffle,
                                                  num_workers=4, drop_last=True)
            for data, labels in batcher:
                yield (data, labels)

    def by_batch(ds, batches=None, bs=1, shuffle=True):
        n = 0
        while True:
            for n, (data, labels) in enumerate(by_epoch(ds, 1, bs, shuffle), n+1) :
                yield (data, labels)
                if n == batches:
                    return

    if epochs is not None:
        return by_epoch(ds, epochs, bs, shuffle)
    else:
        return by_batch(ds, batches, bs, shuffle)

class Callback():
    "Recorder for Trainer class."
    def __init__(self, trainer):
        self.trainer = trainer

    def on_train_begin(self):
        self.tlosses = []
        self.lrs =  []
        self.moms = []

    def on_train_step(self, loss, lr, mom):
        self.tlosses.append(loss)
        self.lrs.append(lr)
        self.moms.append(mom)

    def plotit(self, vals, xlabel, ylabel, ltrim=0):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(ltrim + np.arange(len(vals)), vals)
        ax.grid(True)
        ax.set_ylabel(ylabel)
        ax.set_xlabel(xlabel)

    def plot_lr(self):
        "Plot learning rate schedule"
        self.plotit(self.lrs, "batch", "Learning Rate")

    def plot_mom(self):
        "Plot momentum schedule"
        self.plotit(self.moms, "batch", "Momentum")

    def plot_elr(self):
        "Plot effective learning rate: lr/(1-mom)"
        lr = np.array(self.lrs)
        mom = np.array(self.moms)
        elr = lr / (1 - mom)
        self.plotit(elr, "batch", "Effective Learning Rate")

    def plot_tloss(self, ltrim=0, rtrim=0, tc=0.99):
        "Plot sampled, filtered, and trimmed training loss."
        vals = self.tlosses[ltrim:-rtrim] if rtrim > 0 else self.tlosses[ltrim:]
        f = filter(tc)
        filtered = [f(v) for v in vals]
        self.plotit(filtered, "batch", "Training Loss", ltrim)

    def on_train_end(self):
        pass

class filter():
    "Exponential moving average filter."
    def __init__(self, tc=0.9):
        self.a = None
        self.tc = tc
        pass
    def __call__(self, v):
        if self.a is None:
            self.a = v
        else:
            self.a = self.tc * self.a + (1-self.tc) * v
        return self.a

class ValidationCallback(Callback):
    "Callback for sampling validation loss during training."
    def __init__(self, trainer):
        super().__init__(trainer)
        self.vbatcher = Batcher(self.trainer.validation_set, bs=100)
        self.classifier = Classifier(self.trainer.net)
        self.vlosses = []

    def on_train_step(self, loss, lr, mom):
        super().on_train_step(loss, lr, mom)

        # We sample the validation loss.
        images, labels = next(iter(self.vbatcher))
        logits = self.classifier.logits(images)
        lss = self.trainer.loss(logits, labels.to("cuda")).data.item()
        self.vlosses.append(lss)

    def plot_vloss(self, ltrim=0, rtrim=0, tc=0.99):
        "Plot sampled, filtered, and trimmed validation loss."
        vals = self.vlosses[ltrim:-rtrim] if rtrim > 0 else self.vlosses[ltrim:]
        f = filter(tc)
        filtered = [f(v) for v in vals]
        self.plotit(filtered, "batch", "Validation Loss", ltrim)


class Trainer():
    "Trains a model using datasets, optimizer, and loss."
    def __init__(self, model, train_set, validation_set=None,
                 optimizer=partial(optim.Adam, betas=(0.9, 0.99)),
                 loss=nn.CrossEntropyLoss):

        self.train_set = train_set
        self.validation_set = validation_set
        self.stop_requested = False
        self.net = model.cuda()
        self.loss = loss()
        self.optimizer = optimizer(self.net.parameters(), lr=0.01)

    def request_stop(self):
        "A Callback can request that training stop."
        self.stop_requested = True
        
    def train(self, epochs=None, batches=None, bs=64, lr=0.03, p=0.90, callback=None, silent=False, **kwargs):
        "The training loop--calls out to Callback at appropriate points."

        self.stop_requested = False;
        if callback is None:
            callback = Callback(self)

        batcher = Batcher(self.train_set, epochs, batches, bs, shuffle=True)
        callback.on_train_begin()

        statusbar = tqdm if not silent else lambda x, **args: x
        for i, (batch, labels) in enumerate(statusbar(batcher, total=batches)):

            if self.stop_requested is True:
                print(f"train: stop requested at step {i}")
                break

            learning_rate=lr(i) if callable(lr) else lr
            momentum = p(i) if callable(p) else p

            for param_group in self.optimizer.param_groups:
                param_group['lr'] = learning_rate
                if 'betas' in param_group.keys(): # Adam
                    betas = param_group['betas']
                    param_group['betas'] = (momentum, betas[1])
                elif 'momentum' in param_group.keys(): # Everyone else
                    param_group['momentum'] = momentum
                else:
                    raise KeyError("Cannot find momentum in param_group")

            self.net.train()
            pred = self.net(batch.cuda())

            lss = self.loss(pred, labels.cuda())
            self.optimizer.zero_grad()
            lss.backward()
            self.optimizer.step()

            with torch.no_grad():
                callback.on_train_step(lss.data.item(), learning_rate, momentum)

        callback.on_train_end()
        return callback

class Classifier():
    "Instantiates a model as a classifier."
    def __init__(self, model, device="cuda"):
        self.device = device
        self.model = model.to(device)

    def logits(self, x):
        with torch.no_grad():
            self.model.eval()
            return self.model(x.detach().to(self.device))

    def __call__(self, x):
        "Classifies a batch into predicted labels."
        return torch.argmax(self.softmax(x), 1)

    def softmax(self, x):
        "The probability distribution underlying the prediction."
        return F.softmax(self.logits(x), dim=1)

class VotingClassifier():
    "Combines a set of classifiers into a committee."
    def __init__(self, classifiers):
        self.classifiers = classifiers

    def __call__(self, x):
        "Committee classification of batch."
        with torch.no_grad():
            s = self.softmax(x)
            return torch.argmax(s, 1)

    def softmax(self, x):
        "Combined probability distribution from committee."
        r = 0
        for cl in self.classifiers:
            r += cl.softmax(x)
        return r / len(self.classifiers)

def show_mistakes(classifier, ds, dds=None):
    "Displays mistakes made by a classifier on labeled dataset."
    if dds is None: dds = ds
    mistakes = [m for m in misclassified(classifier, ds)]
    if len(mistakes) == 0:
        print("No mistakes to show!")
        return
    labels = [f"{ds[index][1]} not {pred}" for (index, pred) in mistakes ]
    plot_images(torch.cat([dds[index][0] for (index, _)  in mistakes]), labels=labels)


def misclassified(classifier, ds, bs=100):
    "Iterates through mistakes made by classifier."
    batcher = Batcher(ds, epochs=1, bs=bs, shuffle=False)
    for n, (batch, labels) in enumerate(batcher):
        pred = classifier(batch).cpu()
        mistakes = torch.nonzero(pred != labels)
        for i in mistakes:
            yield (int(i) + n*bs, int(pred[i]) )

def accuracy(classifier, ds, bs=100, include_loss=False):
    "Computes classifer accuracy and optionally loss on a dataset."
    batcher = Batcher(ds, epochs=1, bs=bs)
    correct = tloss = 0

    lossftn = nn.NLLLoss()
    for n, (batch, labels) in enumerate(batcher, 1):
        predictions = classifier(batch).cpu()
        correct += (predictions == labels).sum().item()
        if include_loss:
           logsoftmax = torch.log(classifier.softmax(batch))
           tloss += lossftn(logsoftmax, labels.cuda()).item()

    accuracy = correct / (n * bs)
    loss = tloss / n
    return (accuracy, loss) if include_loss else accuracy

class Residual(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.bn = nn.BatchNorm2d(d)
        self.conv3x3 = nn.Conv2d(in_channels=d, out_channels=d, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.bn(x)
        return x + F.relu(self.conv3x3(x))

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size()[0], -1)

def mnist_model():
    "Returns an initialized but untrained model for MNIST."
    return nn.Sequential(
               nn.Conv2d(in_channels=1, out_channels=128, kernel_size=5, padding=2),
               nn.ReLU(),

               Residual(128),
               nn.MaxPool2d(2),
               Residual(128),

               nn.BatchNorm2d(128),
               nn.Conv2d(128, 256, 3, padding=1),
               nn.ReLU(),
               nn.MaxPool2d(2),
               Residual(256),

               nn.BatchNorm2d(256),
               nn.Conv2d(256, 512, 3, padding=1),
               nn.ReLU(),
               nn.MaxPool2d(2, ceil_mode=True),
               Residual(512),

               nn.BatchNorm2d(512),
               nn.AvgPool2d(kernel_size=4),
               Flatten(),

               nn.Linear(512,10),
               # Softmax provided during training.
           )

def fconcat(f, g, steps):
    return lambda n: f(n) if n < steps else g(n-steps)

def interpolate(a, b, pct):
    return (1-pct)*a + pct*b

def linear_interpolator(a, b, steps):
    return lambda x: interpolate(a, b, x/steps)

def cos_interpolator(a, b, steps):
    def pct(x): return (1 - np.cos(np.pi * x/steps)) / 2
    return lambda x : interpolate(a, b, pct(x))

def exp_interpolator(a, b, steps):
    return lambda x : a*(b/a)**(x/steps)   

def one_cycle(trainer, epochs, bs,
              lr_start, lr_middle, lr_end=None,
              p_start=0.95, p_middle=0.85, p_end=None,
              callback=None, **kwargs):
    "Trains with cyclic learning rate & momentum."

    def schedule(batches, start, middle, end=None):
        if end is None: end=start
        n = int(batches * 0.3)

        f = cos_interpolator(start, middle, n)
        g = cos_interpolator(middle, end, batches - n)
        return fconcat(f, g, n)

    batches = epochs_to_batches(trainer.train_set, epochs, bs)

    p=schedule(batches, p_start, p_middle, p_end)
    lr=schedule(batches, lr_start, lr_middle, lr_end)

    return trainer.train(epochs=None,
                         batches=batches,
                         bs=bs,
                         lr=lr,
                         p=p,
                         callback=callback)

def percent(n):
    return "%.2f%%" % (100 * n)

def show_image(v, title=None):
    "Displays numpy or torch array as image."
    if type(v) is torch.Tensor:
        v = v.numpy()
    v = np.squeeze(v, axis=0)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.imshow(v, cmap="Greys", )

def plot_images(batch, title=None, labels=None):
    "Displays batch of images in a grid."
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
        ax = fig.add_subplot(s, s, i+1)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.axis('off')
        ax.imshow(batch[i,:,:], cmap="Greys")
        if labels is not None:
            ax.annotate(labels[i], xy=(5,0))

def lr_find(trainer, bs, start=1e-6, decades=7, steps=500, p=0.90, **kwargs):
    "Sweep learning rate for model and display loss."

    class LR_Callback(Callback):
        "Callback for implementing lr_find()"
        def __init__(self, trainer):
            super().__init__(trainer)
            self.step = 0
            self.best = math.inf

        def on_train_step(self, loss, lr, mom):
            super().on_train_step(loss, lr, mom)
            self.step += 1
            if (loss > 4 * self.best) and (loss > 2) and (self.step > 50):
                self.trainer.request_stop()
            self.best = min(self.best, loss)

    recorder = LR_Callback(trainer)

    lr = exp_interpolator(start, start*10**decades, steps)
    trainer.train(epochs=None, batches=steps, bs=bs, lr=lr, p=p, callback=recorder, silent=True, **kwargs)

    def plotit(rates, losses, xlabel):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.semilogx(rates, losses)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Loss")
        ax.grid(True)

    plotit(recorder.lrs, recorder.tlosses, "Learning Rate")

    rates = np.array(recorder.lrs)
    eff_rates = rates / (1 - p)
    plotit(eff_rates, recorder.tlosses, "Eff. Learning Rate")

    return recorder

def img_normalize(t):
    c, x, y = t.shape
    t = t.view(c, -1)
    t = t - t.mean(dim=1, keepdim=True)
    t = t / t.std(dim=1, keepdim=True, unbiased=False)
    return t.view(c, x, y)


def create_mnist_datasets(heldout=0, randomize=False):

    rotate_distort = Augmentor.Pipeline()
    rotate_distort.random_distortion(1.0, 4, 4, magnitude=1)
    rotate_distort.rotate(probability=1.0, max_left_rotation=10, max_right_rotation=10)  # probably good

    cropsize = 25
    noncentered_crops = Augmentor.Pipeline()
    noncentered_crops.crop_by_size(1.0, cropsize, cropsize, centre=False)
    noncentered_crops.resize(1.0, 28, 28)

    augmented = transforms.Compose([
                          noncentered_crops.torch_transform(),
                          rotate_distort.torch_transform(),
                          transforms.ToTensor(),
                          transforms.Lambda(img_normalize)
                       ])


    centered_crops = Augmentor.Pipeline()
    centered_crops.crop_by_size(1.0, cropsize, cropsize, centre=True)
    centered_crops.resize(1.0, 28, 28)

    nonaugmented  = transforms.Compose([
                          centered_crops.torch_transform(),
                          transforms.ToTensor(),
                          transforms.Lambda(img_normalize)
                       ])


    train_au = datasets.MNIST('./data', train=True,  download=True, transform=augmented)
    train_na = datasets.MNIST('./data', train=True,  download=True, transform=nonaugmented)
    test_set = datasets.MNIST('./data', train=False, download=True, transform=nonaugmented)

    indices = torch.arange(len(train_na))
    if randomize:
        np.random.shuffle(indices)

    if heldout > 0:
        train_set = torch.utils.data.Subset(train_au, indices[:-heldout])
        valid_set = torch.utils.data.Subset(train_na, indices[-heldout:])
    else:
        train_set = train_au
        valid_set = None

    return (train_set, valid_set, test_set)

def save_model(model, filename):
    "Write the parameters of a model to a file."
    sd = model.state_dict()
    torch.save(sd, filename)

def read_model(model, filename):
    "Read model parameters from file."
    model.load_state_dict(torch.load(filename))
    model.eval()
    return model

####


# def to_onehot(target, nlabels):
#     return torch.eye(nlabels)[target]


class FullCrossEntropyLoss(nn.Module):
    "Cross-entropy loss which takes either class labels or softmax dist. as target."
    def __init__(self):
        super().__init__()

    def forward(self, input, target):
        if target.dim() == 1:
            nlabels = input.size()[-1]
            one_hot = torch.eye(nlabels)[target].to(input.device)
            target = one_hot
        return  -(F.log_softmax(input, 1) * target).sum()

