"""
Created on Tue Feb 23 17:27:01 2021

@author: jonat
"""

"""
Neural Style image network in PyTorch
=====================================
**Implemented by**: `Jonathan Drake <https://github.com/Jonathan-07>`_
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from PIL import Image
from glob import glob
import matplotlib.pyplot as plt

import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.utils import save_image

import copy

# Choose device if you have cuda available (GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


""" Loading images """

# Output image size
imsize = 512 if torch.cuda.is_available() else 128  # Smaller if no GPU

loader = transforms.Compose([
    transforms.Resize(imsize),  # Scale image
    transforms.ToTensor()])  # Transform to Torch tensor


# Function to load image
def image_loader(image_name):
    image = Image.open(image_name)
    image = loader(image).unsqueeze(0)  # Add dummy 'batch' dimension to fit network
    return image.to(device, torch.float)


# Generalise for any image name
style_file = glob('.\\images\\style\\*.*')
content_file = glob('.\\images\\content\\*.*')

style_img = image_loader(style_file[0])
content_img = image_loader(content_file[0])

assert style_img.size() == content_img.size(), \
    "style and content images must be of the same size"

# Display images to check them
unloader = transforms.ToPILImage()  # Reconvert into PIL image

plt.ion()  # Interactive mode on


def imshow(tensor, title=None):
    image = tensor.cpu().clone()  # Clone to avoid editing tensor
    image = image.squeeze(0)  # Remove dummy dimension
    image = unloader(image)
    plt.imshow(image)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # Pause to update plots


plt.figure()
imshow(style_img, title='Style Image')

plt.figure()
imshow(content_img, title='Content Image')


""" Loss functions """

# Content Loss - content distance as transparent layer inserted into network
class ContentLoss(nn.Module):

    def __init__(self, target, ):
        super(ContentLoss, self).__init__()
        # 'Detach' the target content from the tree used
        # to dynamically compute the gradient: this is a stated value,
        # not a variable. Elsewise the forward method of the criterion
        # would throw an error.
        self.target = target.detach()

    def forward(self, input):
        self.loss = F.mse_loss(input, self.target)  # Mean square error
        return input


# Style loss - style distance as transparent layer inserted into network
def gram_matrix(input):
    a, b, c, d = input.size()  # Where a = batch size of 1,
    # b = no. of feature maps,
    # (c,d) = dim of f. map, N=c*d
    features = input.view(a * b, c * d)  # Resize F_xl into \hat F_xl

    G = torch.mm(features, features.t())  # Gram product
    # Norm. Gram matrix values
    return G.div(a * b * c * d)  # By div by no. elements of each F map


# Style loss module same as content, except with Gram matrix
class StyleLoss(nn.Module):

    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        self.target = gram_matrix(target_feature).detach()

    def forward(self, input):
        G = gram_matrix(input)
        self.loss = F.mse_loss(G, self.target)
        return input


"""
Importing model 

Pre-trained 19 layer VGG network - module in 2 child 'Sequential' modules.
'Features' module used --- need output of individual conv layers to measure
content & style loss.
"""

# Set network to eval mode
cnn = models.vgg19(pretrained=True).features.to(device).eval()

# VGG networks trained on images with the following mean & std
cnn_norm_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
cnn_norm_std = torch.tensor([0.229, 0.224, 0.225]).to(device)

# Module for norm of input image, for easy placement in nn.Sequential
class Norm(nn.Module):

    def __init__(self, mean, std):
        super(Norm, self).__init__()
        # View mean & std to make them [C x 1 x 1] to work with image [B x C x H x W]
        self.mean = torch.tensor(mean).view(-1, 1, 1)
        self.std = torch.tensor(std).view(-1, 1, 1)

    def forward(self, img):
        return (img - self.mean) / self.std  # Normalise img


# Create new sequential module with content & style loss layers inserted
# immediately after the desired depth layers they are computing with
content_layers_default = ['conv_4']
style_layers_default = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']


def get_style_model_and_losses(cnn, norm_mean, norm_std,
                               content_img, style_img,
                               content_layers=content_layers_default,
                               style_layers=style_layers_default):
    cnn = copy.deepcopy(cnn)

    # Norm module
    norm = Norm(norm_mean, norm_std).to(device)

    # List of content/style losses
    content_losses = []
    style_losses = []

    # Make new nn.Sequential to which we add modules to be activated sequentially
    model = nn.Sequential(norm)

    i = 0  # Increment for each conv
    for layer in cnn.children():
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = 'conv_{}'.format(i)
        elif isinstance(layer, nn.ReLU):
            name = 'relu_{}'.format(i)
            # ReLU inplace doesn't work with Content & Style-Loss
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = 'pool_{}'.format(i)
        elif isinstance(layer, nn.BatchNorm2d):
            name = 'bn_{}'.format(i)
        else:
            raise RuntimeError('Unrecognised layer: {}'.format(layer.__class__.__name__))

        model.add_module(name, layer)  # Fill in empty model with layers from cnn

        # Add content loss
        if name in content_layers:
            target = model(content_img).detach()
            content_loss = ContentLoss(target)
            model.add_module("content_loss_{}".format(i), content_loss)
            content_losses.append(content_loss)

        # Add style loss
        if name in style_layers:
            target_feature = model(style_img).detach()
            style_loss = StyleLoss(target_feature)
            model.add_module("style_loss_{}".format(i), style_loss)
            style_losses.append(style_loss)

    # Now trim off layers proceeding final content/style losses
    for i in range(len(model) - 1, -1, -1):  # Counts down from 'model length-1' to 0
        if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
            break

    model = model[:(i+1)]

    return model, style_losses, content_losses


"""
Gradient Descent & Running Function

Use L-BFGS optimiser algorithm. Train input image to minimise content/style loss.
Optim requires “closure” function, which reevaluates the module and returns the loss.

For each iteration of the networks, Run func is fed an updated input and computes new losses.
Run backward methods of each loss module to dynamically compute their gradients.
"""

# Create PyTorch L-BFGS optimiser and pass image to it as tensor to optim
def get_input_optim(input_img):
    optimiser = optim.LBFGS([input_img.requires_grad_()])  # Input is param that requires grad
    return optimiser


#Run function
def run_style_transfer(cnn, norm_mean, norm_std, content_img, style_img, input_img,
                       num_steps=600, style_weight=1000000, content_weight=1):

    plt.figure()
    imshow(input_img, title='Input Image')

    print('Building the style transfer model...')
    model, style_losses, content_losses = get_style_model_and_losses(cnn,
                                            norm_mean, norm_std, content_img, style_img)
    optimiser = get_input_optim(input_img)

    print('Optimising...')
    run = [0]
    output_list = [[],[],[]]

    def closure():
        # Correct values in updated input_img to 0<x<1 to keep w/in tensor range
        input_img.data.clamp_(0, 1)

        optimiser.zero_grad()
        model(input_img)
        content_score = 0
        style_score = 0

        for cl in content_losses:
            content_score += cl.loss
        for sl in style_losses:
            style_score += sl.loss

        # Weight loss towards content or style
        content_score *= content_weight
        style_score *= style_weight

        loss = content_score + style_score
        loss.backward()

        # Print loss info every 50 runs
        run[0] += 1
        if run[0] % (num_steps / 10) == 0:
            print("run_{}".format(run))
            print('Content Loss: {:4f} Style Loss: {:4f}'.format(
                content_score.item(), style_score.item()))
            print()
            output_list[0].append(input_img.data.clamp_(0, 1))
            output_list[1].append(content_score.item())
            output_list[2].append(style_score.item())

        return content_score + style_score

    while run[0] <= num_steps:  # Back prop up to num_steps times
        optimiser.step(closure)

    #Final correction
    input_img.data.clamp_(0, 1)

    return input_img, output_list


"""Run Model"""

num_steps = 300


output_weighting_list = []
for x in range(1,10+1):
    # Select input image
    input_img_file = content_img.clone()  # Copy of content img
    # input_img_whitenoise = torch.randn(content_img.data.size(), device=device)  # Or white noise in stead

    output, output_list = run_style_transfer(cnn, cnn_norm_mean, cnn_norm_std,
                                content_img, style_img, input_img_file,
                                content_weight=1, style_weight=1*10**x, num_steps=num_steps)


    fig = plt.figure(figsize=(20, 10))
    ax = []
    for i in range(len(output_list[0])):
        ax.append( fig.add_subplot(4, 5, i + 1) )
        ax[-1].set_title('Content Loss: {:.3f} \n Style Loss: {:.3f}'.format(
                    output_list[1][i], output_list[2][i]))
        plt.ioff()
        imshow(output_list[0][i])
    plt.savefig('./output/progress_{}steps_{}weight.png'.format(num_steps, x))
    plt.close(fig)

    save_image(output[0], './output/output_{}steps_{}weight.png'.format(num_steps, x))