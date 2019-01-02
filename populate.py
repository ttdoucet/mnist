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


    tset, vset, testset = create_mnist_datasets(heldout=0, randomize=False)

    filenames = [ f"model{n+1}.pt" for n in range(npop) ]
    for filename in filenames:
        if os.path.isfile(filename) == False:
            print(f"TRAINING: {filename}")

            trainer = Trainer(mnist_model(), tset, vset)
            cb = one_cycle(trainer, plot=False, **params)

            acc, lss = accuracy_t(Classifier(trainer.net), ds=testset, lossftn=nn.CrossEntropyLoss())
            print(f"TEST: loss = {lss:.3g}, accuracy = {percent(acc)}")
            save_model(trainer.net, filename)

    print("That's all folks!")

def main():
    parser = argparse.ArgumentParser(description='MNIST training utility')
    parser.add_argument('--population', type=int, default=5,
                        help="number of nets in the population")

    args = parser.parse_args()
    populate(args.population)



if __name__ == '__main__':
    main()

