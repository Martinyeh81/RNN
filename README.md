# word sentiment

## Data

The data is from deeplearning.ai (https://www.coursera.org/learn/nlp-sequence-models)

### Dataset:

||X|Y|
| --- | --- | --- |
|0|French macaroon is so tasty|4|
|1|work is horrible|3|
|2|I am upset|3|
|3|throw the ball|1|

![sign1](https://github.com/Martinyeh81/CNN/blob/main/images/amer_sign3.png)

### 400,001 words with 50-dimensional GloVe embeddings:

The Text Corpus is from GloVe: Global Vectors for Word Representation (https://nlp.stanford.edu/projects/glove/)

||the|of|...|sandberger|
| --- | --- | --- | --- | --- |
|0|0.418|0.70853|...|0.072617|
|1|0.24968|0.57088|...|-0.51393|
|2|-0.41242|-0.4716|...|0.4728|
|3|0.1217|0.18048|...|-0.52202|
|:|:|:|:|:|
|49|0.10216|-0.38097|...|0.55559|

![sign2](https://github.com/Martinyeh81/CNN/blob/main/images/american_sign_language.png)

### Number of calsses

![sign3](https://github.com/Martinyeh81/CNN/blob/main/images/number_classes.png)

## Model

trainset's shape is (24709, 784)

Valset's shape is (2746, 784)

testset's shape is (27455, 784)

Compute the cross-entropy cost function J:

$$ J = - \frac{1}{m}  \sum_{i = 1}^m  \large ( \small y^{(i)} \log a^{(i)} + (1-y^{(i)})\log (1-a^{ (i)} )\large )\small$$

1. Simulated the DNN model(epoch = 1500, batch_size = 32, optimizer= adam): LINEAR -> RELU -> LINEAR -> RELU -> LINEAR -> SOFTMAX

![sign4](https://github.com/Martinyeh81/CNN/blob/main/images/DNN_layer.png)

2. CNN model(epoch = 200, batch_size = 64, optimizer= adam): CONV2D -> RELU -> MAXPOOL -> CONV2D -> RELU -> MAXPOOL -> FLATTEN -> FULLYCONNECTED

![sign6](https://github.com/Martinyeh81/CNN/blob/main/images/CNN_layer.png)

3. ResNet model (epoch = 10, batch_size = 32, optimizer= adam):

The identity block is for the case where the input diimension is the same as the output dimension. Also, it skips over 3 hidden laters

![sign6](https://github.com/Martinyeh81/CNN/blob/main/images/idblock3_kiank.png)

The convolutional block is for the case where the input diimension is different from the output dimension. For example, To reduce the dimensions' H and W by a factor of 2, we use a 1x1 convolution with stride 2. Also, it skips over 3 hidden laters

![sign7](https://github.com/Martinyeh81/CNN/blob/main/images/convblock_kiank.png)

![sign8](https://github.com/Martinyeh81/CNN/blob/main/images/Resnet.png)

## Conclusion

||DNN|CNN|ResNet|
| --- | --- | --- | --- |
|Epoch|1500|200|10
|Train Accuracy|1.000|0.921|0.999|
|Val Accuracy|1.000|0.915|1.000|
|Train loss|0.000113|0.235623|0.0011|

DNN Loss function

![sign9](https://github.com/Martinyeh81/CNN/blob/main/images/DNN_loss.png)

CNN Loss function

![sign10](https://github.com/Martinyeh81/CNN/blob/main/images/CNN_loss.png)

ResNet Loss function

![sign11](https://github.com/Martinyeh81/CNN/blob/main/images/ResNet_loss.png)

## Reference

"Sequence Modelss" [Online]. Available: https://www.coursera.org/learn/nlp-sequence-models


