# ISWC22-HAR
This is the code for our paper TinyHAR: A Lightweight Deep Learning Model Designed for Human Activity Recognition ( [ISWC](http://www.iswc.net) 2022 Best Paper Award)

Sensor streams can only be represented in abstract ways and the recorded data typically cannot be interpreted easily by humans. This problem leads to difficulties in post-hoc annotation, which limits the availability and size of annotated Hunman Activity Recognition (HAR) datasets. Given the complexity of sensor-based HAR tasks, such large datasets are typically necessary to apply SOTA machine learning. Although Deep Learning (DL) models have shown extraordinary performance on HAR tasks, most DL models for HAR have large sizes (numerous trainable network parameters). When available data is limited, overly large network parameters make the model prone to overfitting, limiting or even jeopardizing its generalization performance. The second challenge arises from the fact that wearable devices that are intended to use the HAR model typically have limited resources. As a result, an excessive number of network parameters complicates the deployment of such models on end devices. 

To address these challenges, it is desirable to design an efficient and lightweight deep learning model. By reviewing related work, we found only few works that considered designing a lightweight HAR model. To this end, we propose an efficient and lightweight DL model which has small model size and low inference latency.
  
[Cite as](https://publikationen.bibliothek.kit.edu/1000150216)

```
TinyHAR: A Lightweight Deep Learning Model Designed for Human Activity Recognition
Zhou, Y.; Zhao, H.; Huang, Y.; Hefenbrock, M.; Riedel, T.; Beigl, M.
2022. International Symposium on Wearable Computers (ISWC’22) , Atlanta, GA and Cambridge, UK, September 11-15, 2022, Association for Computing Machinery (ACM). doi:10.1145/3544794.3558467 
```

# How to use

TBD

# Network Design ([see paper for details](https://publikationen.bibliothek.kit.edu/1000150216))

![network design](https://user-images.githubusercontent.com/566485/190351410-a32f0056-be32-486a-9410-f87e46a435cd.png)


Designing an optimal, lightweight DL model requires careful consideration of the characteristics of target tasks and the factors which could reduce the inference time and operations number. Based on these two considerations, we developed the following guidelines to design lightweight HAR models:

* **G1**: The Extraction of local temporal context should be enhanced. 
* **G2**: Different sensor modalities should be treated unequally.
* **G3**: Multi-modal fusion.
* **G4**: Global temporal information extraction
* **G5**: The temporal dimension should be reduced appropriately
* **G6**: Channel management, from shallow to deep

## Individual Convolutional Subnet

To enhance the local context, we applied a convolutional subnet to extract and fuse local initial features from the raw data (**G1**). Considering the varying contribution of different modalities, each channel is separately processed through four individual convolutional layers (**G2**). For each convolutional layer, ReLU nonlinearities and batch normalization~\cite{batchnorm} are used. Individual convolution means that the kernels have only 1D structure along the temporal axis (the kernel size is ${5\times1 }$). To reduce the temporal dimension (**G5**), the stride in each layer is set to $2$. All four convolutional layers have the same number of filters $F$. The output shape of this convolutional subnet is thus ${\mathbb{R}^{T^* \times C \times F}}$, where ${T^*}$ denotes the reduced temporal length. 

## Transformer encoder: Cross-Channel Info Interaction

Previous work [1] successfully adopted self-attention mechanism to learn the collaboration between sensor channels. Inspired by this, we utilized one transformer encoder block~\cite{model:transformer} to learn the interaction, which is performed across the sensor channel dimension (**G2**) at each time step. The transformer encoder block consists of a scaled dot-product self-attention layer and a two-layers Fully Connected (FC) feed-forward network. The scaled dot-product self-attention is used to determine relative importance for each sensor channel by considering its similarity to all the other sensor channels. Subsequently, each sensor channel utilized these relative weights to aggregate the features from all the other sensor channels. 

Then the feed-forward layer is applied to each sensor channel, which further fused the aggregated feature of each sensor channel. Until now, the features of each channel are contextualized with the underlying cross-channel interactions. %After this stage, the shape of the data remains the same.

## Fully Connected Layer: Cross-Channel Info Fusion
In order to fuse the learned features from all sensor channels (**G3**), we first vectorize these representations at each time step, ${\vX \in \mathbb{R}^{T^* \times C \times F} \  to \  \vX \in \mathbb{R}^{T^* \times CF}}$. Then one FC layer is applied to weighted summation of all the features. Compared to the attention mechanism used in~\cite{model:attnsense}, in which the features of same sensor channel share the same weights, FC layer allows different features of same sensor channel to have different weights. Such flexibility of the FC layer leads to more sufficient feature fusion. This FC layer works also as a bottleneck layer in the proposed TinyHAR, which reduce the feature dimension to ${F^*}$. In our experiments we set ${F^* = 2F}$.

## One-Layer LSTM: Global Temporal Info Extraction
After the features are fused across sensor and filter dimension, we obtain a sequence of refined feature vectors ${\in \mathbb{R}^{T^* \times F^*}}$ ready for sequence modeling. We then apply one LSTM layer to learn the global temporal dependencies.

## Temporal Attention: Global Temporal Info Enhancement
Given that not all time steps equally contribute to recognition of the undergoing activities, it is crucial to learn the relevance of features at each time step in the sequence. Following the work in~\cite{model:attnsense}, we generate a global contextual representation  ${\vc \in \mathbb{R}^{F^*}}$ by taking a weighted average sum of the hidden states (features) at each time step. The weights are calculated through a temporal self-attention layer. Because the feature at the last time step ${\vx_{T^*} \in \mathbb{R}^{F^*}}$ has the representation for the whole sequence, the generated global representation ${\vc}$ is then added to the ${\vx_{T^*}}$. Here, we introduce a trainable multiplier parameter ${\gamma}$ to ${\vc}$, which allows the model has the ability to flexibly decide, whether to use or discard the generated global representation ${\vc}$.

# Experimental results

[Please refer to our paper for details](https://publikationen.bibliothek.kit.edu/1000150216)

![results](https://user-images.githubusercontent.com/566485/190351810-d762c5db-4ac4-45fd-ab15-d14125f2b0e5.png)

