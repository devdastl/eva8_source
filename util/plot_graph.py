import matplotlib.pyplot as plt
import os

#function to plot loss & accuracy graph.
def plot_loss_accuracy(*argv):

    fig, ax = plt.subplots(2,1, figsize=(8,8))
    
    for indx, args in enumerate(argv):
        loss = args.test_losses
        lbl = "Model" + str(indx+1)
        ax[0].plot(loss, label=lbl)
    ax[0].set_xlabel('Iteration')
    ax[0].set_ylabel('Loss')
    ax[0].legend()

    for indx, args in enumerate(argv):
        acc = args.test_acc
        lbl = "Model" + str(indx+1)
        ax[1].plot(acc, label=lbl)
    ax[1].set_xlabel('Iteration')
    ax[1].set_ylabel('Accuracy')
    ax[1].legend()    

    plt.show()