from mnist import *


def foo(npop, committee):
    _, _, testset = create_mnist_datasets(heldout=0, randomize=False)

    filenames = [ f"model{n+1}.pt" for n in range(npop) ]

    accs = []

    for i in range(npop // committee):

        subset = filenames[i*committee : (i+1)*committee]
        classifiers = [ Classifier(read_model(mnist_model(), filename)) for filename in subset ]

        n = len(classifiers)
        voter_s = VotingSoftmaxClassifier(classifiers)
        acc = accuracy_t(voter_s, ds=testset, bs=100, lossftn=None)

        print(f"{i+1}: Softmax committee of {n} accuracy: {percent(acc)}")
        # show_mistakes(voter_s, testset, testset_na)
        accs.append(acc)

    print(f"mean: {percent(np.mean(accs))} ({np.mean(accs)})" )
    if len(accs) > 1:
        print(f"stdev: {percent(np.std(accs, ddof=1))} ({np.std(accs, ddof=1)})" )


def main():
    _, _, testset = create_mnist_datasets(heldout=0, randomize=False)

    npop = 100

    filenames = [ f"model{n+1}.pt" for n in range(npop) ]

    perm = np.arange(npop)
    accs = []

    for i in range(10):
        np.random.shuffle(perm)

        subset = [filenames[k] for k in perm[:35]]
        classifiers = [ Classifier(read_model(mnist_model(), filename)) for filename in subset ]

        n = len(classifiers)
        voter_s = VotingSoftmaxClassifier(classifiers)
        acc = accuracy_t(voter_s, ds=testset, bs=100, lossftn=None)

        print(f"{i+1}: Softmax committee of {n} accuracy: {percent(acc)}")
        # show_mistakes(voter_s, testset, testset_na)
        accs.append(acc)

    print(f"mean: {percent(np.mean(accs))} ({np.mean(accs)})" )
    print(f"stdev: {percent(np.std(accs, ddof=1))} ({np.std(accs, ddof=1)})" )


if __name__ == '__main__':
    foo(321, 25)

