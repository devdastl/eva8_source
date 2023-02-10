# transforms for training and testing is defined here
#- mean: [tensor(0.4942), tensor(0.4851), tensor(0.4504)]
#- std: [tensor(0.2467), tensor(0.2429), tensor(0.2616)]
# total channel mean - c1+c2+c3/3 = 0.4765

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import numpy as np

mean = [0.4942,0.4851,0.4504]
std = [0.2467,0.2429,0.2616]

def data_albumentation(horizontalflip_prob,rotate_limit,shiftscalerotate_prob,num_holes,cutout_prob):
    # Calculate mean and std deviation for cifar dataset
    
    # Train Phase transformations
    train_transforms = A.Compose([A.HorizontalFlip(p=horizontalflip_prob),
                                  A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=rotate_limit, p=shiftscalerotate_prob),
                                  A.CoarseDropout(max_holes=num_holes,min_holes = 1, max_height=16, max_width=16, 
                                  p=cutout_prob,fill_value=tuple([x * 255.0 for x in mean]),
                                  min_height=16, min_width=16),
                                  A.ColorJitter(p=0.25,brightness=0.3, contrast=0.3, saturation=0.30, hue=0.2),
                                  A.ToGray(p=0.15),
                                  A.Normalize(mean=mean, std=std,always_apply=True),
                                  ToTensorV2()
                                ])

    # Test Phase transformations
    test_transforms = A.Compose([A.Normalize(mean=mean, std=std, always_apply=True),
                                 ToTensorV2()])

    return train_transforms, test_transforms