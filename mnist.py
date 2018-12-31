import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
import itertools
import matplotlib.pyplot as plt
import math
import os
import Augmentor
from  functools import partial

def epochs_to_batches(ds, epochs, bs):
    return epochs * (len(ds) // bs)

def Batcher(ds, epochs=None, batches=None, bs=1, shuffle=True):

    def by_epoch(ds, epochs=1, bs=1, shuffle=True):
        for epoch in range(epochs):
            batcher = torch.utils.data.DataLoader(ds, batch_size=bs, shuffle=shuffle,
                                                  num_workers=4, drop_last=True)
            for data, labels in batcher:
                yield (data, labels)

    def by_batch(ds, batches=None, bs=1, shuffle=True):
        n = 1
        while True:
            for n, (data, labels) in enumerate(by_epoch(ds, 1, bs, shuffle), n) :
                yield (data, labels)
                if n == batches:
                    return

    if epochs is not None:
        return by_epoch(ds, epochs, bs, shuffle)
    else :
        return by_batch(ds, batches, bs, shuffle)

class Callback():
    def __init__(self, learner):
        self.learner = learner
    def on_train_begin(self):
        pass
    def on_train_step(self, step, loss, rate, mom, xs, ys):
        pass
    def on_train_end(self):
        pass
    
class ClassifierCallback(Callback):
    def __init__(self, learner):
        self.learner = learner

    def on_train_begin(self):
        print("training begins")
        self.losses = []
        self.rates =  []

    def accuracy_report(self, dataset, tag, bs):
        acc, lss = accuracy_t(Classifier(self.learner.net), dataset, lossftn=self.learner.loss)
        print(f"{tag}: loss = {lss:.3g}, accuracy = {percent(acc)}")

    def report(self):
        if self.learner.validation_set is not None:
            self.accuracy_report(self.learner.validation_set,
                                 "  validation", bs=50)
        
    def on_train_step(self, step, loss, rate, mom, xs, ys, report_every):

        def accuracy(predictions, actual):
            errors = np.where(predictions != actual)[0]
            return 1.0 - len(errors) / len(actual)

        self.losses.append(loss)
        self.rates.append(rate)
        step += 1

        if (step == 1) or (step % report_every == 0):
            acc = accuracy( (Classifier(self.learner.net)(xs)).cpu().numpy(), ys.numpy())
            print(f"batch {step}: loss = {loss:.3g}, lr = {rate:.3g}, p = {mom:.3g}, accuracy = {percent(acc)}")

        if (step % (10 * report_every) == 0):
            self.report()

    def on_train_end(self):
        print("training ends")

class filter():
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

class AugmentedCallback(ClassifierCallback):
    def __init__(self, learner):
        self.learner = learner
        super().__init__(learner)
        self.vbatcher = Batcher(self.learner.validation_set, bs=100)
        self.classifier = Classifier(self.learner.net)

    def on_train_begin(self):
        super().on_train_begin()
        self.vlosses = []
        self.acc = []

    def on_train_step(self, step, loss, rate, mom, xs, ys, report_every):
        super().on_train_step(step, loss, rate, mom, xs, ys, report_every)

        # We sample the validation loss
        images, labels = next(iter(self.vbatcher))
#        logits = self.classifier.logits(images.to("cuda"))
        logits = self.classifier.logits(images)
        ce_loss = self.learner.loss(logits, labels.to("cuda")).data.item()
        self.vlosses.append(ce_loss)

    def on_train_end(self):
        def plotit(losses, xoffset):
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.plot(xoffset + np.arange(len(losses)), losses)
            ax.grid(True)
            ax.set_ylabel("validation loss")
            ax.set_xlabel("step")
            plt.show()

        f = filter(0.99)
        filtered = [f(vloss) for vloss in self.vlosses]
        half = len(filtered) // 2
        plotit(filtered[:half], 0)
        plotit(filtered[half:], half)
        super().on_train_end()

class Trainer():

    def __init__(self, model, train_set, validation_set,
                 optimizer=partial(optim.Adam, betas=(0.9, 0.99)) ):  #, loss=nn.CrossEntropyLoss):
        self.train_set = train_set
        self.validation_set = validation_set
        self.stop_requested = False
        self.net = model.to("cuda")
        self.loss = nn.CrossEntropyLoss()
        self.optimizer = optimizer(self.net.parameters(), lr=0.01)
#       self.optimizer = optim.Adam(self.net.parameters(), lr=0.01, eps=1e-8, betas=(0.90, 0.99), weight_decay=0)
#        self.optimizer = optim.SGD(self.net.parameters(), lr=0.01, momentum=0, weight_decay=0)

    def request_stop(self):
        self.stop_requested = True
        
    def train(self, epochs=None, batches=None, bs=64, lr=0.03, p=0.90, report_every=100, callback=None, **kwargs):

        self.stop_requested = False;
        if callback is None:
            callback = ClassifierCallback(self)

        if epochs is not None:
            print(f"epochs: {epochs}, batch size: {bs}, batches: {epochs_to_batches(self.train_set, epochs, bs)}")
        else:
            print(f"batch size: {bs}, batches: {batches}")

        batcher = Batcher(self.train_set, epochs, batches, bs, shuffle=True)

        callback.on_train_begin()

        for i, (xs, ys) in enumerate(batcher):
            if self.stop_requested is True:
                print(f"train: stop requested at step {i}")
                break

            learning_rate=lr(i) if callable(lr) else lr
            momentum = p(i) if callable(p) else p

            for param_group in self.optimizer.param_groups:
                param_group['lr'] = learning_rate
                if 'betas' in param_group.keys():
                    # specific to Adam
                    betas = param_group['betas']
                    param_group['betas'] = (momentum, betas[1])
                elif 'momentum' in param_group.keys():
                    param_group['momentum'] = momentum
                else:
                    raise KeyError("Cannot find momentum in param_group")

            batch = xs.to("cuda")
            labels = ys.to("cuda")

            self.net.train()
            pred = self.net(batch)
            ce_loss = self.loss(pred, labels)
            self.optimizer.zero_grad()
            ce_loss.backward()
            self.optimizer.step()

            with torch.no_grad():
                callback.on_train_step(i, ce_loss.data.item(), learning_rate, momentum, xs, ys, report_every)

        callback.on_train_end()
        return callback

class Classifier():
    def __init__(self, model, device="cuda"):
        self.device = device
        self.model = model.to(device)

    def logits(self, x):
        with torch.no_grad():
            self.model.eval()
            return self.model(x.to(self.device))

    def __call__(self, x):
        return torch.argmax(self.softmax(x), 1)

    def softmax(self, x):
        return F.softmax(self.logits(x), dim=1)

class VotingSoftmaxClassifier():
    def __init__(self, classifiers, classes=10):
        self.classifiers = classifiers
        self.classes = classes

    def __call__(self, x):
        with torch.no_grad():
            s = self.softmax(x)
            return torch.argmax(s, 1)

    def softmax(self, x):
        r = torch.zeros([x.shape[0], self.classes]).cuda()
        x = x.cuda()
        for cl in self.classifiers:
            r += cl.softmax(x)
        r /= len(self.classifiers)
        return r

def show_mistakes(classifier, ds, dds=None):
    if dds is None: dds = ds
    indices, preds, truth = misclassified(classifier, ds, bs=25)
    labels = [f"{truth[i]} not {preds[i]}" for i in range(len(preds))]
    plot_images(torch.cat([dds[v][0] for v in indices]), labels=labels)
    plt.show()

# This seems really amateur.
def misclassified(classifier, ds, bs=100):
    batcher = Batcher(ds, epochs=1, bs=bs, shuffle=False)
    wrongs = torch.LongTensor([]).cuda()
    preds = []
    for n, (batch, labels) in enumerate(batcher):
        batch = batch.to("cuda")
        labels = labels.to("cuda")
        pred = classifier(batch).to("cuda")
        idx = torch.nonzero(pred != labels).cuda()
        gidx = n*bs + idx
        for s in [int(xx) for xx in pred[idx]]:
            preds.append(s)
        wrongs = torch.cat((wrongs, gidx), 0)
    indices = wrongs.squeeze().tolist()
    truth = [int(ds[i][1]) for i in indices]
    return indices, preds, truth

def accuracy_t(classifier, ds, bs=100, lossftn=None):
    batcher = Batcher(ds, epochs=1, bs=bs)
    correct = tloss = 0
    for n, (batch, labels) in enumerate(batcher, 1):
        predictions = classifier(batch).cpu()
        correct += (predictions == labels).sum().item()
        if lossftn is not None:
           logits = classifier.logits(batch)
           tloss += lossftn(logits, labels.cuda()).item()
    accuracy = correct / (n * bs)
    loss = tloss / n
    return accuracy if lossftn is None else (accuracy, loss)

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

               nn.Dropout(p=0.20),

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

def elr(lr, p):
    return lambda x: lr(x) / (1-p(x))

def to_p(elr, lr):
    return lambda x: 1 - (lr(x) / elr(x))

def to_lr(elr, p):
    return lambda x: elr(x) * (1-p(x))

def plotfunc(f, steps, label=None):
    x = [f(i) for i in range(steps)]
    plt.plot(x)
    if label is not None:
        plt.ylabel(label)
        plt.xlabel("batch")
    plt.grid(True)
    plt.show()

def one_cycle(learner, epochs, lrmin, lrmax, bs, pmax=0.95, pmin=0.85, 
              lrmin2=None, pmax2=None, callback=None, plot=False, **kwargs):

    def schedule(batches, left, middle, right=None):
        if right is None: right=left
        n = int(batches * 0.3)

        f = cos_interpolator(left, middle, n)
        g = cos_interpolator(middle, right, batches - n)
        return fconcat(f, g, n)

    batches = epochs_to_batches(learner.train_set, epochs, bs)

    p=schedule(batches, left=pmax, middle=pmin, right=pmax2)
    lr=schedule(batches, left=lrmin, middle=lrmax, right=lrmin2)

    if plot:
        plotfunc(lr, batches, "LR")
        plotfunc(p, batches, "P")
        elrf = elr(lr, p)

        plotfunc(elrf, batches, "ELR")
        cumsum = np.cumsum([elrf(i) for i in range(batches)])
        plt.plot(cumsum) 
        plt.ylabel("Sum ELR")
        plt.grid(True)
        plt.show()

        print("lr at end is", lr(batches-1))
        print("elr at end is", elrf(batches-1))

    return learner.train(epochs=None,
                         batches=batches,
                         bs=bs,
                         lr=lr,
                         p=p,
                         callback=callback,
                         report_every=batches//10)

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

def plot_images(batch, title=None, labels=None):
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
        plt.xticks([])
        plt.yticks([])
        plt.axis('off')
        plt.imshow(batch[i,:,:], cmap="Greys")
        if labels is not None:
            plt.annotate(labels[i], xy=(5,0))


def lr_find(learner, bs, batches=500, start=1e-6, decades=7, steps=2500, p=0.90, **kwargs):
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

    lr = exp_interpolator(start, start*10**decades, steps)
    learner.train(epochs=None, batches=steps, bs=bs, lr=lr, p=p, callback=recorder, **kwargs)

    def plotit(rates, losses, xlabel):
        plt.semilogx(rates, losses)
        plt.xlabel(xlabel)
        plt.ylabel("Loss")
        plt.grid(True)
        plt.show()

    plotit(recorder.rates, recorder.losses, "Learning Rate")

    rates = np.array(recorder.rates)
    eff_rates = rates / (1 - p)
    plotit(eff_rates, recorder.losses, "Eff. Learning Rate")

    return recorder

def img_normalize(t):
    c, x, y = t.shape
    t = t.view(c, -1)
    t = t - t.mean(dim=1, keepdim=True)
    t = t / t.std(dim=1, keepdim=True, unbiased=False)
    return t.view(c, x, y)

def create_mnist_datasets(heldout=0, randomize=False):

    p = Augmentor.Pipeline()
    p.random_distortion(1.0, 4, 4, magnitude=1)
    p.rotate(probability=1.0, max_left_rotation=10, max_right_rotation=10)  # probably good

    cropsize = 25

    q = Augmentor.Pipeline()
    q.crop_by_size(1.0, cropsize, cropsize, centre=False)
    q.resize(1.0, 28, 28)

    augmented = transforms.Compose([
                          q.torch_transform(),
                          p.torch_transform(),
                          transforms.ToTensor(),
                          transforms.Lambda(img_normalize)
                       ])


    qq = Augmentor.Pipeline()
    qq.crop_by_size(1.0, cropsize, cropsize, centre=True)
    qq.resize(1.0, 28, 28)

    nonaugmented  = transforms.Compose([
                          qq.torch_transform(),
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
    sd = model.state_dict()
    #for param_tensor in sd:
    #    print(param_tensor, "\t", sd[param_tensor].size())
    torch.save(sd, filename)

def read_model(model, filename):
    model.load_state_dict(torch.load(filename))
    model.eval()
    return model

def ReadClassifier(filename):
    model = read_model(mnist_model(), filename)
    return Classifier(model)

def experiment():
    nonaugmented  = transforms.Compose([
                          transforms.ToTensor(),
                          transforms.Lambda(img_normalize)
                       ])
    testset_na = datasets.MNIST('./data', train=False, download=True, transform=nonaugmented)
    tset, vset, testset = create_mnist_datasets(heldout=0, randomize=False)

    npop = 1
    trainers = [Trainer(mnist_model(), tset, vset) for i in range(npop)]

    lr_eff = 0.001
    pmax = 0.95
    pmin = 0.60
    pmax2 = 0.60

    params = {'epochs': 10, 'bs': 100,
              'lrmin': (1.0 - pmax) * lr_eff,
              'lrmax': (1.0 - pmin) * lr_eff,
              'lrmin2': (1.0 - pmax2) * (lr_eff / 25),
              'pmax' : pmax,
              'pmin' : pmin,
              'pmax2' : pmax2}

    for n, trainer in enumerate(trainers):
        filename = f"model{n+1}.pt"
        if os.path.isfile(filename):
            print("reading", filename)
            read_model(trainer.net, filename)
        else:
            print(f"TRAINING MODEL: {filename}")
            one_cycle(trainer, plot=False, **params)

            acc, lss = accuracy_t(Classifier(trainer.net), ds=testset, lossftn=nn.CrossEntropyLoss())
            print(f"TEST: loss = {lss:.3g}, accuracy = {percent(acc)}")
            save_model(trainer.net, filename)

    classifiers = [Classifier(trainer.net) for trainer in trainers]

    perm = np.arange(npop)
    accs = []

    for i in range(1):
        np.random.shuffle(perm)
        subset = [classifiers[k] for k in perm[:35]]

        voter_s = VotingSoftmaxClassifier(subset, classes=10)
        acc = accuracy_t(voter_s, ds=testset, bs=100, lossftn=None)
        n = len(subset)
        print(f"{i+1}: Softmax committee of {n} accuracy: {percent(acc)}")
        # show_mistakes(voter_s, testset, testset_na)
        accs.append(acc)

    print("mean:", percent(np.mean(accs)), np.mean(accs) )


if __name__ == "__main__":
    experiment()
