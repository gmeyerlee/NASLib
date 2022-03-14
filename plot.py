import numpy as np
import matplotlib.pyplot as plt
import json

def plot_from_json():
    with open('run/nasbench201/cifar10/darts/1000/errors.json') as f:
        errors = json.load(f)

    plt.figure()
    eps = len(errors["train_acc"])
    plt.plot(range(eps), errors["train_acc"], label="train")
    plt.plot(range(eps), errors["valid_acc"], label="val")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig("plots/ipv2_train.png")

def plot_from_slurm():
    train_acc = []
    val_acc = []
    with open('slurm-9422832.out') as f:
        for line in f:
            toks = line.strip().split(" ")
            if len(toks) > 6 and toks[-3] == "accuracy:":
                train_acc.append(float(toks[-6][:-1]))
                val_acc.append(float(toks[-2][:-1]))

    plt.figure()
    eps = len(train_acc)
    plt.plot(range(eps), train_acc, label="train")
    plt.plot(range(eps), val_acc, label="val")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig("plots/evolve_train.png")

def compare_plot(evo, ipevo, title):

    def get_errs(fname):
        if fname[-4:] == ".out":
            train_acc = []
            val_acc = []
            with open(fname) as f:
                for line in f:
                    toks = line.strip().split(" ")
                    if len(toks) > 6 and toks[-3] == "accuracy:":
                        train_acc.append(float(toks[-6][:-1]))
                        val_acc.append(float(toks[-2][:-1]))
            return train_acc, val_acc
        elif fname[-5:] == ".json":
            with open('run/nasbench201/cifar10/darts/'+fname) as f:
                errors = json.load(f)
            return errors["train_acc"], errors["valid_acc"]
        else:
            print("unrecognized error file type")
            return [], []

    ev_train, ev_val = get_errs(evo)
    ip_train, ip_val = get_errs(ipevo)
    plt.figure()
    plt.plot(range(50), ev_train, label="train", color="red")
    plt.plot(range(50), ev_val, label="val", color="gold")
    plt.plot(range(50), ip_train, label="ip_train", color="blue")
    plt.plot(range(50), ip_val, label="ip_val", color="cyan")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.title(title)
    plt.savefig(f"plots/{title}_train.png")

if __name__=="__main__":
    compare_plot('slurm-9422832.out', '1000ipv1/errors.json', 'prev_ip')
    compare_plot('slurm-9422832.out', '1000ipv2/errors.json', 'sample10_select1')
    compare_plot('1000evo2v2/errors.json', '1000ip2v2/errors.json', 'sample10_select2')
    compare_plot('1000evo3/errors.json', '1000ip3/errors.json', 'sample10_select5')


