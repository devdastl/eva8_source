import torch
from src.train import TrainModel
from src.eval import  TestModel
import numpy as np
from torch_lr_finder import LRFinder
import copy
from util.plot_graph import plot_loss_accuracy
from util.plot_misclassified import plot_misclassified, plot_misclassified_grad
from util.get_gradcam import generate_grad

class ModelExecuter():
    
    def __init__(self, net, optimizer, criterion, device, 
                    train_loader, test_loader, save_best=False):
        
        self.net = net
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.save_best = save_best


    def execute_training(self, scheduler=None, NUM_EPOCH=24, use_l1=False, save_best=False):
        self.train_module = TrainModel(self.net, self.device, self.train_loader, self.criterion, self.optimizer, scheduler)
        self.test_module = TestModel(self.net, self.device, self.test_loader, self.criterion)
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

        return self.train_module.lr_trend
        
    def find_lr(self,start_lr=1e-6, end_lr=0.1, plot=False, EPOCH=25, criterion=torch.nn.CrossEntropyLoss()):
         lr_epochs = EPOCH
         num_iter = len(self.test_loader) * lr_epochs
         net = copy.deepcopy(self.net)
         optimizer = self.optimizer
         
         lr_finder = LRFinder(net, optimizer=optimizer, criterion=criterion, device=self.device)
         lr_finder.range_test(self.train_loader, val_loader=self.test_loader, 
                              end_lr=end_lr, num_iter=num_iter, step_mode="linear", diverge_th=50)
         
         max_lr = lr_finder.history['lr'][lr_finder.history['loss'].index(lr_finder.best_loss)]

         if plot:
              lr_finder.plot()

         lr_finder.reset()
         return max_lr
    
    def plot_graph(self):
         plot_loss_accuracy(self.train_module)
         plot_loss_accuracy(self.test_module)

    def plot_misclassified(self):
         plot_misclassified(self.test_module.test_misc_img, self.test_module.test_misc_label, 'model1_misclassified')
         
    def plot_misclassified_grad(self):
         gradcam_img_list = generate_grad(self.test_module.test_misc_img, self.net.to('cpu'))
         plot_misclassified_grad(gradcam_img_list, self.test_module.test_misc_label, 'model1_misclassified_GradCAM')
         





