#function to plot misclassified images.

import matplotlib.pyplot as plt
import torchvision
import numpy as np
import os

def plot_misclassified(test_misc_img, test_misc_label, subtitle='misclassified images'):
# Set the number of rows and columns for the plot
  num_rows = 5
  num_cols = 2

  # Create a figure and axes for the plot
  fig, axes = plt.subplots(num_rows, num_cols, figsize=(8, 8))
  classes= ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

  # Iterate over the mis-classified images and labels
  for i, (img, (pred_label, true_label)) in enumerate(zip(test_misc_img, test_misc_label)):
      # Get the row and column index for the current image
      row = i // num_cols
      col = i % num_cols
      img = img / 2 + 0.5
      npimg = img.to('cpu').numpy()
      axes[row, col].imshow(np.transpose(npimg, (1, 2, 0)))

      # Plot the image and label on the current axes
      #axes[row, col].imshow(img.to('cpu').squeeze(), cmap='gray')
      axes[row, col].set_title(f'Pred: {classes[pred_label]}, True: {classes[true_label]}', fontsize=8)
      # remove axis labels
      axes[row, col].axis('off')
      
  plt.suptitle(subtitle)
  if not os.path.exists("report"):
      os.makedirs("report")
  plt.savefig("report/" + subtitle + ".png")
  plt.show()


def plot_misclassified_grad(test_misc_img, test_misc_label, subtitle='misclassified images'):
# Set the number of rows and columns for the plot
  num_rows = 5
  num_cols = 2

  # Create a figure and axes for the plot
  fig, axes = plt.subplots(num_rows, num_cols, figsize=(8, 8))
  classes= ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

  # Iterate over the mis-classified images and labels
  for i, (img, (pred_label, true_label)) in enumerate(zip(test_misc_img, test_misc_label)):
      # Get the row and column index for the current image
      row = i // num_cols
      col = i % num_cols
      axes[row, col].imshow(img)

      # Plot the image and label on the current axes
      #axes[row, col].imshow(img.to('cpu').squeeze(), cmap='gray')
      axes[row, col].set_title(f'Pred: {classes[pred_label]}, True: {classes[true_label]}', fontsize=8)
      # remove axis labels
      axes[row, col].axis('off')
      
  plt.suptitle(subtitle)
  if not os.path.exists("report"):
      os.makedirs("report")
  plt.savefig("report/" + subtitle + ".png")
  plt.show()