from mnist import *
import argparse

def populate(npop):

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


    tset, vset, _ = create_mnist_datasets(heldout=0, randomize=False)

    filenames = [ f"model{n+1}.pt" for n in range(npop) ]
    for filename in filenames:
        if os.path.isfile(filename) == False:
            print(f"TRAINING: {filename}")

            trainer = Trainer(mnist_model(), tset, vset)
            cb = one_cycle(trainer, plot=False, **params)

            acc, lss = accuracy_t(Classifier(trainer.net), ds=testset, lossftn=nn.CrossEntropyLoss())
            print(f"TEST: loss = {lss:.3g}, accuracy = {percent(acc)}")
            save_model(trainer.net, filename)

    print(f"Population of {npop}: training is complete.")

def run_trials(npop, committee, trials):
    _, _, testset = create_mnist_datasets(heldout=0, randomize=False)

    filenames = [ f"model{n+1}.pt" for n in range(npop) ]

    accs = []
    perm = np.arange(npop)

    for i in range(trials):
        np.random.shuffle(perm)

        subset = [filenames[k] for k in perm[:committee]]

        classifiers = [ Classifier(read_model(mnist_model(), filename)) for filename in subset ]

        n = len(classifiers)

        voter_s = VotingClassifier(classifiers)
        acc = accuracy_t(voter_s, ds=testset, bs=100, lossftn=None)

        print(f"{i+1}: Committee of {n} accuracy: {percent(acc)}")
        # show_mistakes(voter_s, testset, testset_na)
        accs.append(acc)

    print(f"mean: {percent(np.mean(accs))} ({np.mean(accs)})" )


def main():
    parser = argparse.ArgumentParser(description='MNIST training utility')

    parser.add_argument('--population', type=int, default=5,
                        help="number of nets in the population to train")

    parser.add_argument('--committee', type=int, default=5,
                        help="how many nets on a committee")

    parser.add_argument('--trials', type=int, default=10,
                        help='number of committees to form from the population')

    args = parser.parse_args()

    populate(args.population)
    run_trials(args.population, args.committee, args.trials)


if __name__ == '__main__':
    main()

