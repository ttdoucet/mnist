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

def epochs_to_batches(ds, epochs, bs):
    return epochs * (len(ds) // bs)

def by_epoch(ds, epochs=1, bs=1, shuffle=True):
    for epoch in range(epochs):
        batcher = torch.utils.data.DataLoader(ds, batch_size=bs, shuffle=shuffle, num_workers=4, drop_last=True)
        for data, labels in batcher:
            yield (data, labels)

def by_batch(ds, batches=None, bs=1, shuffle=True):
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
#       self.accuracy_report(self.learner.train_set,
#                            "       train", bs=50)

        
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
        #self.report()

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
        self.vbatcher = by_batch(self.learner.validation_set, bs=100)
        self.classifier = Classifier(self.learner.net)

    def on_train_begin(self):
        super().on_train_begin()
        self.vlosses = []
        self.acc = []

    def on_train_step(self, step, loss, rate, mom, xs, ys, report_every):
        super().on_train_step(step, loss, rate, mom, xs, ys, report_every)

        # We sample the validation loss
        images, labels = next(iter(self.vbatcher))
        logits = self.classifier.logits(images.to("cuda"))
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
    def __init__(self, model, train_set, validation_set):
        self.train_set = train_set
        self.validation_set = validation_set
        self.stop_requested = False
        self.net = model.to("cuda")
        self.loss = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.net.parameters(), lr=6.25e-4, eps=1e-8, betas=(0.90, 0.99), weight_decay=0)

    def request_stop(self):
        self.stop_requested = True
        
    def train(self, epochs=1, batches=None, bs=64, lr=0.03, p=0.90, report_every=100, callback=None, **kwargs):

        self.stop_requested = False;
        if callback is None:
            callback = ClassifierCallback(self)

        if epochs is not None:
            print(f"epochs: {epochs}, batch size: {bs}, batches: {epochs_to_batches(self.train_set, epochs, bs)}")
            batcher = by_epoch(self.train_set, epochs, bs, shuffle=True)
        else:
            print(f"batch size: {bs}, batches: {batches}")
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

            with torch.no_grad():
                callback.on_train_step(i, ce_loss.data.item(), learning_rate, momentum, xs, ys, report_every)

        callback.on_train_end()
        return callback

class Classifier():
    def __init__(self, model, device="cuda"):
        self.device = device
        self.model = model.to(device)

    def eval(self, x):
        self.model.eval()
        return self.model(x.detach().to(self.device))

    def __call__(self, x):
        return self.softmax(x).max(1, keepdim=False)[1]

    def logits(self, x):
        return self.eval(x.detach())

    def softmax(self, x):
        return F.softmax(self.eval(x), 1)

class VotingClassifier():
    def __init__(self, classifiers, classes=10):
        self.classifiers = classifiers
        self.classes = classes

    def __call__(self, x):
        r = torch.zeros([x.shape[0], self.classes])
        one_hot = torch.eye(self.classes)
        for cl in self.classifiers:
            r += one_hot[ cl(x) ]
        return r.max(1)[1]

class VotingSoftmaxClassifier():
    def __init__(self, classifiers, classes=10):
        self.classifiers = classifiers
        self.classes = classes

    def __call__(self, x):
        s = self.softmax(x)
        return s.max(1)[1]

    def softmax(self, x):
        r = torch.zeros([x.shape[0], self.classes])
        for cl in self.classifiers:
            r += cl.softmax(x).cpu()
        r /= self.classes
        return r

def show_mistakes(classifier, ds, dds=None):
    if dds is None: dds = ds
    indices, preds, truth = misclassified(classifier, ds)
    labels = [f"{truth[i]} not {preds[i]}" for i in range(len(preds))]
    plot_images(torch.cat([dds[v][0] for v in indices]), labels=labels)
    plt.show()

# This seems really amateur.
def misclassified(classifier, ds, bs=100):
    batcher = by_epoch(ds, epochs=1, bs=bs, shuffle=False)
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
    batcher = by_epoch(ds, epochs=1, bs=bs)
    correct = 0
    tloss = 0
    for n, (batch, labels) in enumerate(batcher, 1):
        batch = batch.to("cuda")
        labels = labels.to("cuda")
        pred = classifier(batch).to("cuda")
        correct += pred.eq(labels.view_as(pred)).sum().item()
        if lossftn is not None:
           logits = classifier.logits(batch)
           tloss += lossftn(logits, labels).item()
    accuracy = correct / (n * bs)
    loss = tloss / n
    return accuracy if lossftn is None else (accuracy, loss)

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

def mnist_model():
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

#  Jeremy stuff, playing around
def conv_layer(ni:int, nf:int, ks:int=3, stride:int=1)->nn.Sequential:
    "Create Conv2d->BatchNorm2d->LeakyReLu layer: `ni` input, `nf` out filters, `ks` kernel, `stride`:stride."
    return nn.Sequential(
        nn.Conv2d(ni, nf, kernel_size=ks, bias=False, stride=stride, padding=ks//2),
        nn.BatchNorm2d(nf),
        nn.LeakyReLU(negative_slope=0.1, inplace=True))

def mnist_jeremy_model():
    def conv(ni,nf): return nn.Conv2d(ni, nf, kernel_size=3, stride=2, padding=1)
    def conv2(ni,nf): return conv_layer(ni,nf,stride=2)
    class res_block(nn.Module):
        def __init__(self, nf):
            super().__init__()
            self.conv1 = conv_layer(nf,nf)
            self.conv2 = conv_layer(nf,nf)

        def forward(self, x): return x + self.conv2(self.conv1(x))

    return nn.Sequential(
        conv2(1, 8),
        res_block(8),
        conv2(8, 16),
        res_block(16),
        conv2(16, 32),
        res_block(32),
        conv2(32, 16),
        res_block(16),
        conv2(16, 10),
        Flatten()
    )

# End of Jeremy stuff.

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

def TTA_Classifier(model, n, classes=10):
    cl = Classifier(model)
    tta = [cl for i in range(n)]
    return VotingSoftmaxClassifier(tta, classes=10)

def trial(trainer, epochs, cycles=1):
    params = {'epochs': epochs, 'bs': 100,
              'lrmin': 5e-5, 'lrmax': 5e-4,
              'pmax' : 0.95, 'pmin' : 0.70, 'pmax2' : 0.70}

    for i in range(cycles):
        cb = one_cycle(trainer, **params)

    return cb

def experiment():
    print("experiment")

    nonaugmented  = transforms.Compose([
                          transforms.ToTensor(),
                          transforms.Lambda(img_normalize)
                       ])

    testset_na = datasets.MNIST('./data', train=False, download=True, transform=nonaugmented)

    tset, vset, testset = create_mnist_datasets(heldout=0, randomize=False)

    npop = 105
    trainers = [Trainer(mnist_model(), tset, vset) for i in range(npop)]

    params = {'epochs': 8, 'bs': 100,
              'lrmin': 5e-5, 'lrmax': 5e-4,
              'pmax' : 0.95, 'pmin' : 0.70, 'pmax2' : 0.70}

    for n, trainer in enumerate(trainers):
        filename = f"model{n+1}.pt"
        if os.path.isfile(filename):
            read_model(trainer.net, filename)
        else:
            print(f"TRAINING MODEL: {filename}")
            # one_cycle(trainer, **params)
            trial(trainer, epochs=5, cycles=1)
            acc, lss = accuracy_t(Classifier(trainer.net), ds=testset, lossftn=nn.CrossEntropyLoss())
            print(f"TEST: loss = {lss:.3g}, accuracy = {percent(acc)}")

            save_model(trainer.net, filename)

#    classifiers = [TTA_Classifier(trainer.net, 10) for trainer in trainers]
    classifiers = [Classifier(trainer.net) for trainer in trainers]

    perm = np.arange(npop)
    accs = []

    for i in range(15):
        np.random.shuffle(perm)
        subset = [classifiers[k] for k in perm[:15]]
#        subset = [ classifiers[i] ]

        voter_s = VotingSoftmaxClassifier(subset, classes=10)
        acc = accuracy_t(voter_s, ds=testset, bs=25, lossftn=None)
        n = len(subset)
        print(f"{i+1}: Softmax committee of {n} accuracy: {percent(acc)}")
        show_mistakes(voter_s, testset, testset_na)
        accs.append(acc)

    print("mean:", percent(np.mean(accs)), np.mean(accs) )

def main():
#    tset, vset, test_set = create_mnist_datasets(heldout=1000, randomize=True)
    tset, vset, test_set = create_mnist_datasets(heldout=0, randomize=False)
    print("training set size is", len(tset))
    print("no holdout, randomize=False, constant pmax")

    trainer = Trainer(mnist_model(), tset, vset)

    params = {'epochs': 8, 'bs': 100,
              'lrmin': 5e-5, 'lrmax': 5e-4,
              'pmax' : 0.95, 'pmin' : 0.70, 'pmax2' : 0.70}

    cb = one_cycle(trainer, **params)

    cl = Classifier(trainer.net)

    accs = []
    samps = 20
    for i in range(samps):
        acc, lss = accuracy_t(cl, ds=test_set, bs=25, lossftn=nn.CrossEntropyLoss())
        print(f"{i+1}: test set: accuracy={percent(acc)}, loss={lss:.3g}")
        accs.append(acc)
    print("average:", np.mean(accs))

    tta = [cl for i in range(samps)]
    voter_tta = VotingSoftmaxClassifier(tta, classes=10)
    acc = accuracy_t(voter_tta, ds=test_set, bs=25, lossftn=None)
    print(f"test set: voting accuracy={percent(acc)}")

    # acc, lss = accuracy_t(Classifier(trainer.net), test_set, lossftn=nn.CrossEntropyLoss() )
    # print(f"test set: loss={lss:.3g}, accuracy={percent(acc)}")

    return cb

if __name__ == "__main__":
    experiment()
