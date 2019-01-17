# Written by Todd Doucet.
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
import math
import Augmentor
from  functools import partial
from fastprogress import progress_bar

def epochs_to_batches(ds, epochs, bs):
    return epochs * (len(ds) // bs)

def Batcher(ds, bs, epochs=None, batches=None, shuffle=True):
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

def perplexity(preds):
    entropy = -(preds.softmax(dim=1) * preds.log_softmax(dim=1)).sum(dim=1)
    perplexity = entropy.exp()
    return perplexity.mean()


class Callback():
    "Recorder for Trainer class."
    def __init__(self, trainer):
        self.trainer = trainer

    def on_train_begin(self):
        self.tlosses = []
        self.lrs =  []
        self.moms = []
        self.logits = []
        self.accs = []

    def on_train_step(self, loss, lr, mom, logits, acc):
        self.tlosses.append(loss)
        self.lrs.append(lr)
        self.moms.append(mom)
        self.logits.append(logits)
        self.accs.append(acc)

    def axes(self, data, filt, dslice):
        filtered = [filt(v) for v in data]
        ys = filtered[dslice]
        xs = np.arange(len(data))[dslice]
        return (xs, ys)

    def batch_plot(self, plots, ylabel, ymax=None, loc="best"):
        fig, ax = plt.subplots()
        for (xs, ys), args in plots:
            ax.plot(xs, ys, **args)
            ax.grid(True)
            ax.set_ylabel(ylabel)
            ax.set_xlabel("batch")
            ax.legend(loc=loc)
            if ymax is not None:
                ax.set_ylim([None, ymax])

    def loss_plot(self, plots, **kwargs):
        self.batch_plot(plots, ylabel="loss", **kwargs)

    def plot_loss(self, start=None, stop=None, step=None, ymax=None, halflife=0):
        "Plot sampled, filtered, and trimmed training loss."
        dslice = slice(start, stop, step)
        plot = [self.axes(self.tlosses, filter(halflife), dslice), {'label' : 'train', 'color': 'C1'} ]
        self.loss_plot( [plot], ymax=ymax)

    def plot_logits(self, start=None, stop=None, step=None, halflife=0):
        "Plot sampled, filtered, and trimmed magnitude of output logits."
        dslice = slice(start, stop, step)
        norms = [ logit.norm(dim=1).mean() for logit in self.logits]
        plot = [self.axes(norms, filter(halflife), dslice), {'label' : 'train', 'color': 'C1'} ]
        self.batch_plot([plot], ylabel="logit magnitude")

    def plot_perplexity(self, start=None, stop=None, step=None, halflife=0):
        "Plot sampled, filtered, and trimmed entropy of predictions."
        dslice = slice(start, stop, step)
        perps = [ perplexity(logit) for logit in self.logits]
        plot = [self.axes(perps, filter(halflife), dslice), {'label' : 'train', 'color': 'C1'} ]
        self.batch_plot([plot], ylabel="perplexity")

    def plot_accuracy(self, start=None, stop=None, step=None, halflife=0):
        "Plot sampled, filtered, and trimmed accuracy of predictions."
        dslice = slice(start, stop, step)
        plot = [self.axes(self.accs, filter(halflife), dslice), {'label' : 'train', 'color': 'C1'} ]
        self.batch_plot([plot], ylabel="accuracy")

    def plot_schedule(self, start=None, stop=None, step=None, plot_mom=True, plot_elr=False):
        "Plot learning rate and momentum schedule."
        dslice = slice(start, stop, step)
        lr = np.array(self.lrs)
        mom = np.array(self.moms)
        elr = lr / (1 - mom)

        fig, ax = plt.subplots()
        if plot_elr:
            elrs = self.axes(elr, filter(0), dslice)
            ax.plot(*elrs, label='ELR', color='C0')

        lrs = self.axes(lr, filter(0), dslice)
        ax.plot(*lrs, label='LR', color='C9')

        if plot_mom:
            ax2 = ax.twinx()
            moms = self.axes(mom, filter(0), dslice)
            ax2.plot(*moms, label='MOM', color='C6')
            ax2.legend(loc='center right')
            ax2.set_ylabel("Momentum", color='C6')

        ax.legend(loc='upper right')
        ax.set_ylabel('Learning rate', color='C9')
        ax.set_xlabel('batch')
        plt.show()


class filter():
    "Exponential moving average filter."
    def __init__(self, halflife=10):
        self.a = None
        self.d = 0 if halflife==0 else 0.5 ** (1/halflife)

    def __call__(self, v):
        self.a = v if self.a is None else self.d * self.a + (1-self.d) * v
        return self.a

class ValidationCallback(Callback):
    "Callback for sampling validation loss during training."
    def __init__(self, trainer, bs=100):
        super().__init__(trainer)
        self.vbatcher = Batcher(self.trainer.validation_set, bs=bs)
        self.vlosses = []
        self.vlogits = []
        self.vaccs = []

    def on_train_step(self, loss, lr, mom, logits, acc):
        super().on_train_step(loss, lr, mom, logits, acc)

        # Sample the validation loss.
        images, labels = next(iter(self.vbatcher))
        vlogits = self.trainer.net(images.cuda())
        vloss = self.trainer.loss(vlogits, labels.cuda()).item()
        self.vlosses.append(vloss)
        self.vlogits.append(vlogits.detach())

        with torch.no_grad():
            correct = (torch.argmax(vlogits, dim=1) == labels.cuda()).sum()
            accuracy = correct.type(dtype=torch.cuda.FloatTensor) / len(labels)
            self.vaccs.append(accuracy)

    def plot_loss(self, start=None, stop=None, step=None,  halflife=0, include_train=True, ymax=None):
        "Plot sampled, filtered, and trimmed validation loss."
        dslice = slice(start, stop, step)
        vplot = [self.axes(self.vlosses, filter(halflife), dslice), {'label' : 'valid', 'color': 'C0'} ]
        tplot = [self.axes(self.tlosses, filter(halflife), dslice), {'label' : 'train', 'color': 'C1'}]
        self.loss_plot( [tplot, vplot] if include_train else [ vplot ] )
        plt.show()

    def plot_perplexity(self, start=None, stop=None, step=None, include_train=True, halflife=0):
        "Plot sampled, filtered, and trimmed entropy of predictions."
        dslice = slice(start, stop, step)

        vperps = [ perplexity(logit) for logit in self.vlogits]
        vplot = [self.axes(vperps, filter(halflife), dslice), {'label' : 'valid', 'color': 'C0'} ]

        tperps = [ perplexity(logit) for logit in self.logits]
        tplot = [self.axes(tperps, filter(halflife), dslice), {'label' : 'train', 'color': 'C1'} ]

        plots = [tplot, vplot] if include_train else [ vplot ]
        self.batch_plot(plots, ylabel="perplexity")

    def plot_accuracy(self, start=None, stop=None, step=None, include_train=True, halflife=0):
        "Plot sampled, filtered, and trimmed accuracy of predictions."
        dslice = slice(start, stop, step)

        vplot = [self.axes(self.vaccs, filter(halflife), dslice), {'label' : 'valid', 'color': 'C0'} ]
        tplot = [self.axes(self.accs, filter(halflife), dslice),  {'label' : 'train', 'color': 'C1'} ]

        plots = [tplot, vplot] if include_train else [ vplot ]
        self.batch_plot(plots, ylabel="accuracy")

def onehot(target, nlabels):
    return torch.eye(nlabels)[target]

class FullCrossEntropyLoss(nn.Module):
    "Cross-entropy loss which takes either class labels or prob. dist. as target."
    def __init__(self):
        super().__init__()

    def forward(self, input, target):
        n, nlabels = input.shape
        if target.dim() == 1:
            target = onehot(target, nlabels).to(input.device)
        return  -(F.log_softmax(input, dim=1) * target).sum() / n 

class Trainer():
    "Trains a model using datasets, optimizer, and loss."
    def __init__(self, model, train_set, validation_set=None,
                 optimizer=partial(optim.Adam, betas=(0.9, 0.99)),
                 loss=FullCrossEntropyLoss):

        self.train_set = train_set
        self.validation_set = validation_set
        self.net = model.cuda()
        self.loss = loss()
        self.optimizer = optimizer(self.net.parameters(), lr=0.01)

    def set_hyperparameters(self, lr, p):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
            if 'betas' in param_group.keys(): # Adam
                betas = param_group['betas']
                param_group['betas'] = (p, betas[1])
            elif 'momentum' in param_group.keys(): # Everyone else
                param_group['momentum'] = p
            else:
                raise KeyError("Cannot find momentum in param_group")

    def train_steps(self, lr, p, epochs=None, batches=None, bs=None,
                    callback=None, authority=None, **kwargs):
        "Iterator interface to the training loop."

        if callback is None:
            callback = Callback(self)

        batcher = Batcher(self.train_set, bs, epochs, batches, shuffle=True)
        callback.on_train_begin()

        for i, (batch, labels) in enumerate(batcher):
            learning_rate=lr(i) if callable(lr) else lr
            momentum = p(i) if callable(p) else p
            self.set_hyperparameters(learning_rate, momentum)

            if authority is not None:
                labels = authority(batch)

            self.net.train()
            logits = self.net(batch.cuda())
            loss = self.loss(logits, labels.cuda())
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            with torch.no_grad():
                correct = (torch.argmax(logits, dim=1) == labels.cuda()).sum()
                accuracy = correct.type(dtype=torch.cuda.FloatTensor) / bs
                callback.on_train_step(loss.detach(), learning_rate, momentum,
                                       logits.detach(), accuracy )

            yield(callback)

    def train(self, lr, p, epochs=None, batches=None, bs=None, **kwargs):
        "The training loop--calls out to Callback at appropriate points"
        cycles = batches if batches is not None else epochs_to_batches(self.train_set, epochs, bs)

        steps = self.train_steps(lr, p, **dict(kwargs, batches=batches, bs=bs, epochs=epochs))
        for step in progress_bar(steps, total=cycles):
            pass
        return step

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
        return torch.argmax(self.softmax(x), dim=1)

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
            return torch.argmax(s, dim=1)

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
    batcher = Batcher(ds, bs, epochs=1, shuffle=False)
    for n, (batch, labels) in enumerate(batcher):
        pred = classifier(batch).cpu()
        mistakes = torch.nonzero(pred != labels)
        for i in mistakes:
            yield (int(i) + n*bs, int(pred[i]) )

def accuracy(classifier, ds, bs=100, include_loss=False):
    "Computes classifer accuracy and optionally loss on a dataset."
    batcher = Batcher(ds, bs, epochs=1)
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
              pct=0.3,
              batches=None, callback=None, yielder=False,
              authority=None,
              **kwargs):
    "Trains with cyclic learning rate & momentum."

    def schedule(batches, start, middle, end=None, pct=0.3):
        if end is None: end=start
        n = int(batches * pct)
        f = cos_interpolator(start, middle, n)
        g = cos_interpolator(middle, end, batches - n)
        return fconcat(f, g, n)

    if batches is None:
        batches = epochs_to_batches(trainer.train_set, epochs, bs)

    p=schedule(batches, p_start, p_middle, p_end, pct)
    lr=schedule(batches, lr_start, lr_middle, lr_end, pct)

    train = trainer.train_steps if yielder else trainer.train
    return train(epochs=None, batches=batches, bs=bs, lr=lr, p=p,
                 authority=authority, callback=callback)

def percent(n):
    return "%.2f%%" % (100 * n)

def show_image(v, title=None, interpolation='bilinear'):
    "Displays numpy or torch array as image."
    if type(v) is torch.Tensor:
        v = v.numpy()
    v = np.squeeze(v, axis=0)
    fig, ax = plt.subplots()
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.imshow(v, interpolation=interpolation, cmap="Greys", )

def plot_images(imgs, labels=None):
    "Displays batch of images in a grid."
    if type(imgs) is torch.Tensor:
        imgs = imgs.cpu().detach().numpy()

    *_, r, c = imgs.shape
    imgs = imgs.reshape(-1, r, c)
    n = imgs.shape[0]
    s = int(math.sqrt(n) + 0.5)
    if (s*s < n):
        s += 1

    fig, ax = plt.subplots(figsize=(6,6))
    for i in range(n):
        ax = plt.subplot2grid( (s,s), (i//s, i%s) )
        ax.set_xticks([])
        ax.set_yticks([])
        ax.axis('off')
        ax.imshow(imgs[i,:,:], interpolation='bilinear', cmap="Greys")
        if labels is not None:
            ax.annotate(labels[i], xy=(5,0))
    plt.show()

def lr_find(trainer, bs, start=1e-6, decades=7, steps=500, p=0.90, plot_elr=False, **kwargs):
    "Sweep learning rate for model and display loss."
    lr = exp_interpolator(start, start*10**decades, steps)
    steps = trainer.train_steps(epochs=None, batches=steps, bs=bs, lr=lr, p=p, **kwargs)
    best = math.inf
    for i, step in enumerate(steps):
        loss = step.tlosses[-1]
        if (loss > 4 * best) and (loss > 2) and (i > 50):
            print(f"stopping at step {i}")
            break
        best = min(best, loss)

    def plotit(rates, losses, xlabel):
        fig, ax = plt.subplots()
        ax.semilogx(rates, losses, color='C1')
        ax.set_xlabel(xlabel)
        ax.set_ylabel("train loss")
        ax.grid(True)
        plt.show()

    plotit(step.lrs, step.tlosses, "learning rate")
    rates = np.array(step.lrs)
    if plot_elr:
        eff_rates = rates / (1 - p)
        plotit(eff_rates, step.tlosses, "eff. learning rate")
        return step

def img_normalize(t):
    c, x, y = t.shape
    t = t.view(c, -1)
    t = t - t.mean(dim=1, keepdim=True)
    t = t / t.std(dim=1, keepdim=True, unbiased=False)
    return t.view(c, x, y)

def save_model(model, filename):
    "Write the parameters of a model to a file."
    sd = model.state_dict()
    torch.save(sd, filename)

def read_model(model, filename):
    "Read model parameters from file."
    model.load_state_dict(torch.load(filename))
    model.eval()
    return model

## MNIST-specific stuff
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

def augmented_pipeline():
    rotate_distort = Augmentor.Pipeline()
    rotate_distort.random_distortion(1.0, 4, 4, magnitude=1)
    rotate_distort.rotate(probability=1.0, max_left_rotation=10, max_right_rotation=10)

    cropsize = 25
    noncentered_crops = Augmentor.Pipeline()
    noncentered_crops.crop_by_size(1.0, cropsize, cropsize, centre=False)
    noncentered_crops.resize(1.0, 28, 28)

    return transforms.Compose([
                          noncentered_crops.torch_transform(),
                          rotate_distort.torch_transform(),
                          transforms.ToTensor(),
                          transforms.Lambda(img_normalize)
                      ])

def nonaugmented_pipeline():
    centered_crops = Augmentor.Pipeline()
    cropsize = 25
    centered_crops.crop_by_size(1.0, cropsize, cropsize, centre=True)
    centered_crops.resize(1.0, 28, 28)

    return   transforms.Compose([
                          centered_crops.torch_transform(),
                          transforms.ToTensor(),
                          transforms.Lambda(img_normalize)
                        ])

def mnist_trainset(heldout=0, randomize=False, augmented=True):
    xform = augmented_pipeline() if augmented else nonaugmented_pipeline()
    train = datasets.MNIST('./data', train=True,  download=True, transform=xform)

    indices = np.arange(len(train))
    if randomize:
        np.random.shuffle(indices)
    if heldout > 0:
        train_set = torch.utils.data.Subset(train, indices[:-heldout])
        valid_set = torch.utils.data.Subset(train, indices[-heldout:])
        return (train_set, valid_set)
    else:
        return torch.utils.data.Subset(train, indices[:])

def mnist_testset(augmented=False):
    xform = augmented_pipeline() if augmented else nonaugmented_pipeline()
    return datasets.MNIST('./data', train=False, download=True, transform=xform)
