# General Adversarial Networks

GANs and DCGANs Implementation
GANs are a framework where we have two players, in terms of Game theory or two
networks working against each other (Min - Max game). The goal is to achieve Nash
Equilibrium where we say that the Generator is producing images that the
discriminator cannot differentiate between.

GANs have been used for a variety of applications and is an active topic for research. In
the implementation of DCGAN, I observed that GANs are highly unstable and can
either losses can quickly become zero i.e one overpowering the other.
There is a lot of tweaking the hyper-parameters involved to ensure that you get the
desired result. There have been a lot of research to make sure of the stability of the
GANs. The Self-Attention conditional GAN explores a couple of methods that can
better the result.

SAGAN Implementation
The main idea of using Self Attention GANs are due to the fact that GANs cannot do
very well on structure and often don't capture shapes very well, that is images tend to
look realistic but out of sorts.

Some of the solutions for this is to increase the convolutional filters receptive field size
so they can capture more information and can relate to the more, this works to a
certain extent but it makes the training of GANs harder and longer.
Attention helps establish the similarity and understand if two features are related.As
mentioned in the paper, DCGAN's really does well in terms of the
background and texture - the skies are usually blue and recognized well but sometimes
the specific object is not clear to see.

<img src ="https://user-images.githubusercontent.com/46288072/75120092-a003c800-5656-11ea-8f1b-ed706d821273.png" width="200" height="200">
Figure 1: Sample from Generator 30K iter.


The first square in the below figure is a frog, but the generator does not do a good job
on the structure of it, but its texture and color are very good.
Self attention calculates long range dependencies, it does this by calculating attention
vectors - which are like hot spots in the image that are most similar too for a particular
region. We do this process for all the positions. There is a choice of where you would
want your attention layer to be inserted - which convolution layer - ideally the attention
layer is implemented in between the 16x16 and 32x32 layers in the Generator.

Attention gives the generator a broader view - and helps it better understand more
complicated features that it might have missed. This enables the generator to create
images of higher detail and structural meaning. The implementation of the
self-attention layer, I follow the paper for the structure of attention.
We use f,g and h which are 1x1 convolutions on the layer where you implement it. f and
g are later passed through a softmax function after a matrix multiplication. The
softmax returns an attention map. It is then combined with h to result the
self-attention feature maps.

Since there is no direct implementation of attention in keras, we must build our custom
layer - attention module as a class and dene our operations and then use the object
created from the class in DCGANs. 􀀀 is a hyperparameter that is optimized, amount of
attention to be applied. This might be possibly implemented using Keras - Lambda
function. The same procedure is true for spectral normalization, most implementations
have created a custom layer and applied it.

Figure 2: Self Attention Module
As mentioned in the paper - there are two ways of stabilizing GANs during training.

Spectral Normalization
It is used for both discriminator and generator. During training generator
networks sometimes fail to learn the multimodal structure of the target , there is a
possibility that there will be a discriminator that can perfectly distinguish target
and generator distribution - this means that the generator has no chance and it's
training comes to stop.
Spectral Normalization controls the Lipschitz constant of the discriminator by
making sure that the spectral norm of every layer is within limits. This method
leads to faster convergence, better quality of images generated. target distribution.

Two Time Scale Update Rule - This is used to address the slow learning in
discriminators, however I did not see the eect of this for DCGANs convergence.
This is simply implemented by changing the learning rates.
Figure 3: Generator(left) and Discriminator Architecture(right) - using keras.plotmodel
30K iter.


<img src="https://user-images.githubusercontent.com/46288072/75120204-ae061880-5657-11ea-8ad2-8effeb26da69.png" width="500" height="500"> 

Figure 4: a) DCGAN - Generated Grid 8x8 
There is a sharp decrease in FID for the rst few thousand iterations and then it is
pretty much constant. The average FID score for the DCGAN for 74; 000 iterations is
292:54


## Implementation Notes

This project, I have implemented DCGAN completely. SAGAN implementation is incomplete. The Jupyter notebook contains the GAN with the training. 

Every `1000` iterations I have saved the Generator Model to the drive, to ensure maximum train time - I have seperated the process of calculating FID score.

I am attaching the [![Python 3.7](https://img.shields.io/badge/python-3.7-blue.svg)](https://www.python.org/downloads/release/python-370/) code which I used to run to generate the images from the generator and then calculate the FID.

## Requirements 

The requirements for this project and all dependencies are there in the `requirements.txt` file.

## Files Included
1) best_model.h5
2) Report
3) Model --> json for architecture
4) Requirements
5) DCGAN and Self-Attention jupyter notebook

