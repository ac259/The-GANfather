from keras import layers, models
import numpy as np
import os, glob
from keras.preprocessing import image
from numpy import zeros
from numpy.random import randn
from matplotlib import pyplot
from time import gmtime, strftime
import numpy
import keras
from numpy import cov
from numpy import trace
from numpy import iscomplexobj
from numpy import asarray
from numpy.random import shuffle
from scipy.linalg import sqrtm
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import preprocess_input
from keras.datasets.mnist import load_data
from skimage.transform import resize
from keras.datasets import cifar10

def scale_images(images, new_shape):
    images_list = list()
    for image in images:
        # resize with nearest neighbor interpolation
        new_image = resize(image, new_shape, 0)
        # store
        images_list.append(new_image)
    return asarray(images_list)


# calculate frechet inception distance
def calculate_fid(model, act1, act2):
    
    # calculate mean and covariance statistics
    mu1, sigma1 = act1.mean(axis=0), cov(act1, rowvar=False)
    mu2, sigma2 = act2.mean(axis=0), cov(act2, rowvar=False)
    # calculate sum squared difference between means
    ssdiff = numpy.sum((mu1 - mu2)**2.0)
    # calculate sqrt of product between cov
    covmean = sqrtm(sigma1.dot(sigma2))
    # check and correct imaginary numbers from sqrt
    if iscomplexobj(covmean):
        covmean = covmean.real
    # calculate score
    fid = ssdiff + trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid

# generate points in latent space as input for the generator
def generate_latent_points(latent_dim, n_samples):
    # generate points in the latent space
    x_input = randn(latent_dim * n_samples)
    # reshape into a batch of inputs for the network
    x_input = x_input.reshape(n_samples, latent_dim)
    return x_input

# use the generator to generate n fake examples, with class labels
def generate_fake_samples(g_model, latent_dim, n_samples):
    # generate points in latent space
    x_input = generate_latent_points(latent_dim, n_samples)
    # predict outputs
    X = g_model.predict(x_input)
    return X

# Imports to load the model
from keras.models import load_model
from keras.models import model_from_json
import json

with open('model_in_json.json','r') as f:
    model_json = json.load(f)

model = model_from_json(model_json)
(_, _), (images2, _) = keras.datasets.cifar10.load_data()
images2 = (images2 - 127.5) / 127.5

path = os.getcwd()
path = os.path.join(path,'GAN_weight_2')
# For all weights in the folder
# Generate the image and calculate the FIDs
for i in glob.glob(path+'/*.h5'):
    model.load_weights(i)

    filename = os.path.split(i)[-1]

    n_samples = 100
    latent_dim = 100
    X = generate_fake_samples(model, latent_dim, n_samples)
    #print(X.shape)
    images1 = X

    model_inception = InceptionV3(include_top=False, pooling='avg', input_shape=(299,299,3))
    # load cifar10 images
    shuffle(images2)
    #   images1
    images2 = images2[:100]
    #print('Loaded', images1.shape, images2.shape)
    # convert integer to floating point values
    images1 = images1.astype('float32')
    images2 = images2.astype('float32')
    # resize images
    images1 = scale_images(images1, (299,299,3))
    images2 = scale_images(images2, (299,299,3))

    # calculate activations
    act1 = model_inception.predict(images1)
    act2 = model_inception.predict(images2)

    # calculate fid
    fid = calculate_fid(model_inception, act1, act2)
    print(f'FID is {round(fid,3)} and generator is {filename}')