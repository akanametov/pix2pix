Pix2Pix
======================

:star: Star this project on GitHub — it helps!

[Pix2Pix](https://arxiv.org/abs/1609.04802) is an example of in GAN's. It allows to map input image
to its different representation. As **Generator** this GAN uses
**UNet** like architecture. The **Discriminator** is simple feed-forward model with **Conv2d** as output.


## Table of content

- [Evaluation](#eval)
- [Database](#database)
- [Training](#train)
- [License](#license)
- [Links](#links)

## Evaluation

You can evaluate pretrained **Generator** of **Pix2Pix** on your images.
To do this use `eval.py`.

### Database

This **Pix2Pix** GAN was trained on **Facades** dataset.

### Training

We train booth **Generator** and **Discriminator** with their loss functions.
The **Generator's loss** consists of **Adverserial loss**(BCE loss between *fake's prediction and fake's target*)
and **Pixel-wise loss**(L1 loss between *fake and real images*). For **Discriminator** the loss function is a sum of
**Adverserial losses**: BCE loss between *fake's prediction and fake's target*, BCE loss between *real's prediction and real's target*.

<a>
    <img src="images/g_loss.png" align="center" height="300px" width="300px"/>
</a>

<a>
    <img src="images/d_loss.png" align="center" height="300px" width="300px"/>
</a>

**After 100 epochs of training:**

<a>
    <img src="images/train.png" align="center" height="500px" width="400px"/>
</a>


See [demo](https://github.com/akanametov/Pix2Pix/blob/main/demo/demo.ipynb) for **Pix2Pix**'s training.

## License

This project is licensed under MIT.

## Links

* [Pix2Pix GAN (arXiv article)](https://arxiv.org/pdf/1611.07004.pdf)
