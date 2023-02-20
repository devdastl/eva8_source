import torch
from src.train import TrainModel
from src.eval import  TestModel
import numpy as np
from torch_lr_finder import LRFinder
from util.plot_graph import plot_loss_accuracy
from util.plot_misclassified import plot_misclassified, plot_misclassified_grad
from util.get_gradcam import generate_grad

class ModelExecuter():
    
    def __init__(self, criterion, device, 
                    train_loader, test_loader, save_best=False):
        
        self.criterion = criterion
        self.device = device
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.save_best = save_best
        self.train_module = None
        self.test_module = None
        self.trained_net = None


    def execute_training(self, net, optimizer, scheduler=None, NUM_EPOCH=24, use_l1=False, save_best=False):
        self.train_module = TrainModel(net, self.device, self.train_loader, self.criterion, optimizer, scheduler)
        self.test_module = TestModel(net, self.device, self.test_loader, self.criterion)
        self.scheduler=scheduler
        self.EPOCH = NUM_EPOCH

        for epoch in range(1, self.EPOCH+1):
                print("EPOCH:", epoch)
                self.train_module.train_a(L1_reg=use_l1, l1_lambda=0.0001) # batch norm model with L1 regularization.
                self.test_module.eval(epoch, self.EPOCH+1)

                # update LR
                if self.scheduler:
                    if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                        self.scheduler.step()    
     
        self.trained_net = net
        return self.train_module.lr_trend
        
    def find_lr(self, net, optimizer, end_lr=10, num_iter=200, step_mode='exp', start_lr=None, diverge_th=5, plot=True):
     lr_finder = LRFinder(net, optimizer, self.criterion, device=self.device)
     lr_finder.range_test(self.test_loader, end_lr=end_lr, num_iter=num_iter, step_mode=step_mode, start_lr=start_lr, diverge_th=diverge_th)
     min_loss = min(lr_finder.history['loss'])
     max_lr = lr_finder.history['lr'][np.argmin(lr_finder.history['loss'], axis=0)]

     print("Min Loss = {}, Max LR = {}".format(min_loss, max_lr))
     # Reset the model and optimizer to initial state

     if plot:
          lr_finder.plot()

     lr_finder.reset()

     return min_loss, max_lr
    
    def plot_graph(self):
         plot_loss_accuracy(self.train_module, title = "accuracy-loss for Training")
         plot_loss_accuracy(self.test_module, title = "accuracy-loss for Evaluation")

    def plot_misclassified(self):
         plot_misclassified(self.test_module.test_misc_img, self.test_module.test_misc_label, 'model1_misclassified')
         
    def plot_misclassified_grad(self):
         gradcam_img_list = generate_grad(self.test_module.test_misc_img, self.trained_net.to('cpu'))
         plot_misclassified_grad(gradcam_img_list, self.test_module.test_misc_label, 'model1_misclassified_GradCAM')
         





