# The-GANfather
Make a model you can't refuse

---

## Ian Goodfellow: Generative Adversarial Networks (NIPS 2016 tutorial)
## Why study generative models?
* Excellent to test of ability to use High-dimensional data
* Simulated RL
* Missing data
* Realistic generation tasks
    * Single Image Super-Resolution
![1_5tuN3sDrbq9ug2BdeTwiQQ](https://user-images.githubusercontent.com/46288072/56601361-e583de80-65c8-11e9-8aff-975c94ae87d8.png)
(Ledig et al 2010)

* Adobe has an interactive GAN - iGAN where a user can draw a line, which would be translated to something - mountain by the generative model.

* Image to Image Translation - edges to photos

---
`What is a Markov Chain? What are Autoencoders?`

There are two types for Explicit Density -> <b>Approximate density</b>
* Variational
    * Variational Autoencoder
* Markov Chain
    * Boltzmann Machine

One example of a fully visible belief net is <i>Wavenet</i>. Amazing quality but it takes two minutes to synthesize one second of audio. (Lot of time)

---

## <u>GANs</u>
* Use latent code
* Asymptotically consistent
* No Markov chains needed
* Often regarded as producing the best samples.

**Training Procedure**

Use SGD - algorithm of choice ADAM optimizer on two minibatches simultaneously.
* A minibatch of training examples
* A minibatch of generated samples
Run gradient descent on both of the players cost functions simultaneously


