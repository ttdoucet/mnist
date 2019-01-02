# Written by Todd Doucet.

from mnist import *
import argparse

def populate(npop):
    "Train a population of neural nets and save them to disk."

    # The one-cycle training schedule to use.
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

    # Our training set, with augmentation.
    tset, _, testset = create_mnist_datasets(heldout=0, randomize=False)

    # The names of the files containing the models' weights.
    filenames = [ f"model{n+1}.pt" for n in range(npop) ]

    # Train and save all the models that have not yet
    # been saved.  If interrupted, this continues where
    # it left off.
    for filename in filenames:
        if os.path.isfile(filename) == False:
            print(f"TRAINING: {filename}")

            # Take a new untrained MNIST model and wrap
            # it in a Trainer.
            trainer = Trainer(mnist_model(), tset, None)

            # Train for one cycle using the parameter specified above.
            one_cycle(trainer, **params)

            # After training, see how this individual net does on the test set.
            acc, lss = accuracy(Classifier(trainer.net), ds=testset, lossftn=nn.CrossEntropyLoss())
            print(f"TEST: loss = {lss:.3g}, accuracy = {percent(acc)}")

            # Write the model to disk.
            save_model(trainer.net, filename)

    print(f"Population of {npop}: training is complete.")


def run_trials(npop, committee, trials):
    "For each trial, form a committee out of the population and classify."
    _, _, testset = create_mnist_datasets(heldout=0, randomize=False)

    filenames = [ f"model{n+1}.pt" for n in range(npop) ]

    accs = []
    perm = np.arange(npop)

    for i in range(trials):

        # Randomly choose 'committee' of the model files from the population.
        np.random.shuffle(perm)
        subset = [filenames[k] for k in perm[:committee]]

        # Read those files into models and create classifiers for them.
        classifiers = [ Classifier(read_model(mnist_model(), filename)) for filename in subset ]

        # Create a voting classifier from this set of individual classifiers.
        voter_s = VotingClassifier(classifiers)

        # See how it does!
        acc = accuracy(voter_s, ds=testset)
        print(f"{i+1}: Committee of {committee} accuracy: {percent(acc)}")

        accs.append(acc)

    print(f"mean: {percent(np.mean(accs))} ({np.mean(accs):.6g})" )


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

