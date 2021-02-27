# Sentences Sentiment

## Data

The data is from deeplearning.ai (https://www.coursera.org/learn/nlp-sequence-models)

### Dataset:

||X|Y|
| --- | --- | --- |
|0|French macaroon is so tasty|4|
|1|work is horrible|3|
|2|I am upset|3|
|3|throw the ball|1|

![sen1](https://github.com/Martinyeh81/RNN/blob/main/images/data_set.png)

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


## Model

trainset's shape is (183, 2)

testset's shape is (53, 2)

Compute the softmax cross-entropy cost function J:

$$ J = - \frac{1}{m}  \sum_{i = 1}^m  \large ( \small y^{(i)} \log a^{(i)}\large )\small$$

### Simulated the RNN model(epoch = 400, optimizer= adam):

![sen2](https://github.com/Martinyeh81/RNN/blob/main/images/image_1.png)

### RNN-LSTM model(epochs = 50, batch_size = 32, optimizer= adam):

![sen3](https://github.com/Martinyeh81/RNN/blob/main/images/emojifier-v2.png)


## Conclusion

||RNN|RNN-LSTM|
| --- | --- | --- |
|Epoch|400|50|
|Train Accuracy|0.923|1.000|
|Val Accuracy|0.892|0.982|
|Train loss|0.3665|0.0024|

### RNN Loss function and Confusion Matrix 

||Precision|Recall|F1-score|
| --- | --- | --- | --- |
|0|1.00|0.75|0.86|
|1|1.00|1.00|1.00|
|2|0.81|0.94|0.87|
|3|0.87|0.87|0.87|
|4|1.00|1.00|1.00|

![sen4](https://github.com/Martinyeh81/RNN/blob/main/images/RNN_loss.png)

### RNN-LSTM Loss function and Confusion Matrix 

||Precision|Recall|F1-score|
| --- | --- | --- | --- |
|0|1.00|0.92|0.96|
|1|1.00|1.00|1.00|
|2|1.00|1.00|1.00|
|3|0.94|1.00|0.97|
|4|1.00|1.00|1.00|

![sen5](https://github.com/Martinyeh81/RNN/blob/main/images/RNN_LSTM_loss.png)


## Reference

"Sequence Modelss" [Online]. Available: https://www.coursera.org/learn/nlp-sequence-models


