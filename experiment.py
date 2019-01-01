from mnist import *

def experiment():
    nonaugmented  = transforms.Compose([
                          transforms.ToTensor(),
                          transforms.Lambda(img_normalize)
                       ])
    testset_na = datasets.MNIST('./data', train=False, download=True, transform=nonaugmented)
    tset, vset, testset = create_mnist_datasets(heldout=0, randomize=False)

    npop = 51
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
            read_model(trainer.net, filename)

            # acc = accuracy_t(Classifier(trainer.net), ds=testset, bs=100, lossftn=None)
            # print(f"{filename}: {percent(acc)}")


        else:
            print(f"TRAINING MODEL: {filename}")
            one_cycle(trainer, plot=False, **params)

            acc, lss = accuracy_t(Classifier(trainer.net), ds=testset, lossftn=nn.CrossEntropyLoss())
            print(f"TEST: loss = {lss:.3g}, accuracy = {percent(acc)}")
            save_model(trainer.net, filename)

    classifiers = [Classifier(trainer.net) for trainer in trainers]

    perm = np.arange(npop)
    accs = []

    for i in range(20):
        np.random.shuffle(perm)
        subset = [classifiers[k] for k in perm[:35]]

        voter_s = VotingSoftmaxClassifier(subset)
        acc = accuracy_t(voter_s, ds=testset, bs=100, lossftn=None)
        n = len(subset)
        print(f"{i+1}: Softmax committee of {n} accuracy: {percent(acc)}")
        # show_mistakes(voter_s, testset, testset_na)
        accs.append(acc)

    print("mean:", percent(np.mean(accs)), np.mean(accs) )



experiment()
