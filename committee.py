# Written by Todd Doucet.
from mnist import *
import argparse
import os
import sys

# probably to the library
def annihilate(cb, epochs=1):
    lr_start = cb.lrs[-1]
    batches = epochs_to_batches(cb.trainer.train_set, epochs, cb.bs)
    lr = cos_interpolator(lr_start, lr_start/1e3, batches)
    return cb.trainer.train(lr=lr, p=cb.moms[-1], wd=cb.wd, epochs=epochs, bs=cb.bs, callback=cb)

def train_one(trainset, testset):
    epochs, lr_max, bs = (10, 4e-4, 100)

    model = mnist_model()
    trainer = Trainer(model, trainset)
    lr = exp_interpolator(lr_max, lr_max/10, epochs_to_batches(trainset, epochs, bs))
    cb = trainer.train(lr, p=0.90, epochs=epochs, bs=bs)
    annihilate(cb)

    acc, lss = accuracy(Classifier(model), ds=testset, include_loss=True)
    print(f"TEST: loss = {lss:.3g}, accuracy = {percent(acc)}")
    return model

def populate(npop):
    "Train a population of neural nets and save them to disk."

    # Our training set, with augmentation.
    trainset = mnist_trainset(heldout=0)
    testset = mnist_testset()

    # The names of the files containing the models' weights.
    filenames = [ f"model{n+1}.pt" for n in range(npop) ]

    # Train and save all the models that have not yet
    # been saved.  If interrupted, this continues where
    # it left off.
    for filename in filenames:
        if os.path.isfile(filename) == False:
            print(f"TRAINING: {filename}")
            model = train_one(trainset, testset)
            save_model(model, filename)

    print(f"Population of {npop}: training is complete.")

def run_trials(npop, committee, trials):
    "For each trial, form a committee out of the population and classify."
    testset = mnist_testset()

    filenames = [ f"model{n+1}.pt" for n in range(npop) ]

    accs = []
    perm = np.arange(npop)
    np.random.shuffle(perm)

    for trial in range(trials):
        start = trial * committee;
        subset = [filenames[k] for k in perm[start : start + committee]]

        # Read those files into models and create classifiers for them.
        classifiers = [ Classifier(read_model(mnist_model(), filename)) for filename in subset ]

        # Create a voting classifier from this set of individual classifiers.
        voter_s = VotingClassifier(classifiers)

        # See how it does!
        acc = accuracy(voter_s, ds=testset)
        print(f"{trial+1} of {trials}: Committee of {committee} accuracy: {percent(acc)}")
        accs.append(acc)

    print(f"mean: {percent(np.mean(accs))} ({np.mean(accs):.6g})" )


def main():
    parser = argparse.ArgumentParser(description='MNIST training utility')

    parser.add_argument('--population', type=int, default=5,
                        help="number of nets in the population to train")

    parser.add_argument('--committee', type=int, default=5,
                        help="how many nets on a committee")

    parser.add_argument('--trials', type=int, default=None,
                        help='number of committees to form from the population')

    args = parser.parse_args()
    max_trials = args.population // args.committee

    trials = max_trials if args.trials is None else args.trials
    if trials > max_trials:
        print(f"Max trials for {args.population} population and {args.committee} committee size: {max_trials}",
              file=sys.stderr)
        sys.exit(-1)
    
    populate(args.population)
    run_trials(args.population, args.committee, trials)

if __name__ == '__main__':
    main()

