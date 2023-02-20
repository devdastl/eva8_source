import matplotlib.pyplot as plt
import os

#function to plot loss & accuracy graph.
def plot_loss_accuracy(*argv, title="None"):

    fig, ax = plt.subplots(2,1, figsize=(8,8))
    
    for indx, args in enumerate(argv):
        loss = args.losses
        lbl = "Model" + str(indx+1)
        ax[0].plot(loss, label=lbl)
    ax[0].set_xlabel('Iteration')
    ax[0].set_ylabel('Loss')
    ax[0].legend()

    for indx, args in enumerate(argv):
        acc = args.acc
        lbl = "Model" + str(indx+1)
        ax[1].plot(acc, label=lbl)
    ax[1].set_xlabel('Iteration')
    ax[1].set_ylabel('Accuracy')
    ax[1].legend()    

    plt.suptitle(title)
    plt.show()