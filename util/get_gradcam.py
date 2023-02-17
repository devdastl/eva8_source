import numpy as np
from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

def generate_grad(images, model):
    input_tensor_list = [img.unsqueeze(0).to('cpu') for img in images]
    rgb_img_list = []
    grad_img_list = []

    for rgb_img in input_tensor_list:
        rgb_img = rgb_img.squeeze(0).to('cpu').numpy()
        rgb_img = (rgb_img - np.min(rgb_img)) / (np.max(rgb_img) - np.min(rgb_img))
        rgb_img = np.transpose(rgb_img, (1, 2, 0))
        rgb_img_list.append(rgb_img)

    target_layer4 = [model.layer3[-1]]
    cam = GradCAM(model=model, target_layers=target_layer4, use_cuda=False)

    for input_tensor, input_rgb in zip(input_tensor_list, rgb_img_list):
        grayscale_cam4 = cam(input_tensor=input_tensor)
        grayscale_cam4 = grayscale_cam4[0, :]
        grad_img = (show_cam_on_image(input_rgb, grayscale_cam4, use_rgb=True))
        grad_img_list.append(grad_img)


    return grad_img_list