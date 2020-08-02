Code for my undergraduate Thesis "Novel Approaches for Neural Text Representation and Generation"
==============================================================================================

## 0. Outline

The objective of the thesis was to explore and evaluate potential novel approaches for representing and generating natural text with neural network based methods. Throughout the work many different methods have been implemented and tested. Some of them were fairly new and yet not found in literature (e.g. hierarchical discrete representation of text) while others were improvements over previous work or the application of methods already used in other problem domains like image generation to the field of text generation.

After experimenting with many different models, the focus was put on exploring new GAN architectures and training approaches for text generation. The architectural novelty is in the the application of U-Net [1] based discriminators [2] in which the discriminator does not just predicts a single value for the "fakeness" of a given sample, but does so for every atomic value of the sample which equals to tokens in the text setting. This approach has been both evaluated for generating text in a continuous space by employing pre-trained word embeddings as a intermediate representation layer as well as with discrete representations by using the Boundary Seeking GAN architecture [3]. Especially first model turned out to be highly promising, although state-of-the-art results could not be reached yet.

Apart from the aforementioned approaches and models, the following methods have been implemented and evaluated in the course of this work too but have been turned down due to unpromising results, being too difficult to be handled in the small scope of the thesis or due to lack of computational resources:

1. Hierarchical text representations using VQ-VAEs [4] for text [5], [6] and training different generative models on top of such representations. Among those generative models that have been evaluated are
    1. Auto-regressive models, like RNNs and the self-attention based Transformer architecture
    1. (Multi-scale) [7], [8] GANs [9], [10]
1. Training of progressively growing GANs using the Transformer architecture [11] both for the generator and discriminator
1. Partially-invertible GANs for implicitly learning an invertible continuous representation of input discrete data 

## 1. Adversarial Text Generation with U-Net based Discriminators

### 1.1 Overview
Adversarial approaches for text generation have only lead to modest results so far [10], [12], [13], [14]. Among the reasons for this is the difficulty of training GANs with discrete data as well as the impossibility to apply novel and highly promising architectures and training schemes like multi-stage GANs to text data.

One approach, that has proven to outperform any multi-stage based GAN model, which are usually also fairly difficult to implement, is the application of U-Net based discriminators [2]. While the discriminator classically gives a "fakeness" prediction for the sample as a whole, in this model the U-Net architecture, commonly known from image segmentation problems in the biomedical field, is used for the discriminator to additionally predict scores for every single value in the sample. Latter corresponds to pixels in the image and tokens in the text domain.

By doing so, the path between the discriminator and generator can be shortened which leads to more meaningful error gradients for the generator.

As an additional side-effect, the discriminator is trained to perform the discrimination task on token level (e.g. characters or words). Upon successful training, it could be used to perform different side-tasks like spelling detection or detecting wrong information (bad spelling or wrong text sections would likely be rejected as "fake").

#### 1.1.1 Text Representations

While GANs can be applied to image data more or less in a trivial way, for text data respectively discrete data careful choices need to made about the architecture, loss function as well for the representation of the data itself.

In particular, two ways of employing GANs for text generation are common:
1. Training on discrete data using gradient approximation techniques (e.g. REINFORCE [13], [14], Boundary Seeking GAN [3])
1. Training on continuous representations of discrete data (e.g. word2vec [16] for text [15])

While the discrete approach allows the generator to model the data in a more natural way, it usually gives hard constraints on the choice of architecture, makes the training more difficult, unstable and oftentimes leads to bad results.

On the other hand, using continuous representations of discrete data, for example by using word2vec representations in the case of text, gives more freedom on the choice of architecture and loss functions but there is little work about whether such continuous approximations are guaranteed to lead to good results in the GAN setting.

To evaluate the performance of the U-Net based discriminator for text generation, we tried the approach in the discrete setting by employing a Boundary Seeking GAN [3] as well as in the continuous data by using pre-trained word2vec representations. In latter case, the continuous output of the generator is turned into a discrete representation by a nearest-neighbor search in the embedding space.

### 1.2 Running the Code

The code to run both the discrete and continuous model is provided with this repository.

Before running, the requirements should be installed by executing the following command within the repository folder:

```bash
pip install -r requirements.txt
```

#### 1.2.1 Download Data

The data mainly used for the experiments consists of sentence n-grams created from articles from the German SPIEGEL Online news site.

The dataset can be obtained [here](https://drive.google.com/file/d/1uZ6NmODpT9bQ_xFTULiRYv8GZfNtSTiJ/view?usp=sharing).

#### 1.2.2 Run Discrete Model

The discrete model, based on the Boundary Seeking GAN can be trained by running

```bash
python train_unet_bgan.py --dataset <path-to-text-file>
```

where `<path-to-text-file>` is the path to a text file containing one sample per line. For the experiments, the `unigrams/train.txt` file from the previously mentioned dataset has been used.

#### 1.2.3 Run Continuous Model

The continuous model can be trained using the following command:

```bash
python train_unet_word_vec_gan.py <path-to-text-file>
```

Again `<path-to-text-file>` must correspond to a text file containing one sample per line. `bigrams/large/train.txt` has been used for most of the experiments.

The given data is first tokenized to a fixed sized vocabulary using Byte Pair Encoding before it is mapped into a continuous representation using pre-trained word vectors. For both tasks, the BPEmb [17] library is employed.

#### 1.2.4 Hyperparameters

Both models offer a variety of hyperparameters, e.g. the sequence length, learning rate, etc. These can be mostly controlled using command-line options offered by the training scripts.

The available command line options can be displayed by running either script with the `--help` option.

### 1.3 Results

Coming soon...


## 1.4 Legacy Experiments

Coming soon...


## 1.5 References

[1] Olaf Ronneberger, Philipp Fischer, and Thomas Brox. (2015). U-Net: Convolutional Networks for Biomedical Image Segmentation.

[2] Edgar Schönfeld, Bernt Schiele, and Anna Khoreva. (2020). A U-Net Based Discriminator for Generative Adversarial Networks.

[3] R Devon Hjelm, Athul Paul Jacob, Tong Che, Adam Trischler, Kyunghyun Cho, and Yoshua Bengio. (2017). Boundary-Seeking Generative Adversarial Networks.

[4] Aaron van den Oord, Oriol Vinyals, & Koray Kavukcuoglu. (2017). Neural Discrete Representation Learning.
[5] Łukasz Kaiser, & Samy Bengio. (2018). Discrete Autoencoders for Sequence Models.

[6] Łukasz Kaiser, Aurko Roy, Ashish Vaswani, Niki Parmar, Samy Bengio, Jakob Uszkoreit, & Noam Shazeer. (2018). Fast Decoding in Sequence Models using Discrete Latent Variables.

[7] Tero Karras, Timo Aila, Samuli Laine, & Jaakko Lehtinen. (2017). Progressive Growing of GANs for Improved Quality, Stability, and Variation.

[8] Animesh Karnewar, & Oliver Wang. (2019). MSG-GAN: Multi-Scale Gradients for Generative Adversarial Networks.

[9] Ian J. Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, & Yoshua Bengio. (2014). Generative Adversarial Networks.

[10] Martin Arjovsky, Soumith Chintala, & Léon Bottou. (2017). Wasserstein GAN.

[11] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, & Illia Polosukhin. (2017). Attention Is All You Need.

[12] Jiaxian Guo, Sidi Lu, Han Cai, Weinan Zhang, Yong Yu, & Jun Wang. (2017). Long Text Generation via Adversarial Training with Leaked Information.

[13] Lantao Yu, Weinan Zhang, Jun Wang, & Yong Yu. (2016). SeqGAN: Sequence Generative Adversarial Nets with Policy Gradient.

[14] Cyprien de Masson d'Autume, Mihaela Rosca, Jack Rae, & Shakir Mohamed. (2019). Training language GANs from Scratch.

[15] Budhkar, A., Vishnubhotla, K., Hossain, S., & Rudzicz, F. (2019). Generative Adversarial Networks for Text Using Word2vec
 IntermediariesProceedings of the 4th Workshop on Representation Learning for NLP (RepL4NLP-2019).

[16] Tomas Mikolov, Kai Chen, Greg Corrado, & Jeffrey Dean. (2013). Efficient Estimation of Word Representations in Vector Space.

[17] Benjamin Heinzerling, & Michael Strube. (2017). BPEmb: Tokenization-free Pre-trained Subword Embeddings in 275 Languages.
