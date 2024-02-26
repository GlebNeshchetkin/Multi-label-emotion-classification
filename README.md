# Multi-label-emotion-classification
*The model aims to improve performance of multi-label emotions classification using sentiment classification. The experimental results demonstrate that the learning approach can achieve high level of performance on two reference data sets.\
Two datasets were used for training and testing: Twitter dataset from **SemEval 2016 Task 4A** (Nakov et al., 2016) as a source for sentiment classification task and Twitter dataset **SemEval 2018 Task 1C** (Mohammad et al., 2018) which contains 11 emotions for target task emotion classification.*
## <kbd style="background-color: #0074cc; color: #fff; border-radius: 50%; padding: 4px 8px;">1</kbd>  Sentiment Classification with BERT
Sentiment classification model is a PyTorch-based implementation of a sentiment classification model using BERT architecture. The BERT model is loaded from the 'bert-base-uncased' pretrained model.  \
The classifier consists of a feed-forward neural network with multiple hidden layers, batch normalization, ReLU activation functions, and dropout layers. The input to the classifier is the last hidden state of the special token [CLS] obtained from the BERT model. The number of labels for the classification task is specified by the num_labels=3 parameter (POSITIVE, NEITHER, NEGATIVE).
### The model architecture can be summarized as follows:
- BERT Model: Loaded from 'bert-base-uncased', with the option to freeze its parameters (freeze_bert parameter).
- Feed-Forward Classifier: Composed of 6 fully connected layers with batch normalization, ReLU activation, and dropout layers.
- Last Linear Layer: The final layer of the classifier, producing logits for each class.
### Training loop
Training loop for a sentiment classification model utilizes CrossEntropyLoss, Adam optimizer, and a MultiStepLR scheduler for learning rate adjustment. The loop iterates over 40 epochs, updating model parameters on a provided dataset.
## <kbd style="background-color: #0074cc; color: #fff; border-radius: 50%; padding: 4px 8px;">2</kbd> Emotion classification model
Emotion classification model is a PyTorch implementation of a text classification model named BertClassifierEm. The model utilizes the BERT pre-trained model and weights of last layer of Sentiment classification model.
### The model architecture can be summarized as follows:
- BERT Model: Loaded from 'bert-base-uncased'.
- The BERT output is fed into a LSTM layer, adding a sequential aspect to the model to capture temporal dependencies.
- The LSTM output + Last layer of Sentiment classificator then passed through a classifier composed of 3 linear layers with layer normalization, leaky ReLU activations, and dropout for regularization.
### Training loop
Training loop for sentiment classification model utilizes MultiLabelSoftMarginLoss (weighted), Adam optimizer, and a MultiStepLR scheduler for learning rate adjustment. The loop iterates over 10 epochs, updating model parameters on a provided dataset.
## Results
The model was compared with SpanEmo and DATN-1 (Dual Attention Transfer Network) results.
### Model without boosting (M1) versus Model with boosting (M2)
| Model     |  ACC  |   F1  |
|-----------|-------|-------|
|    M1     | 0.629 | 0.366 |
|    M2     | 0.659 | 0.479 |

### Comparison of different models
| Model     |  ACC  |   F1  |
|-----------|-------|-------|
| SpanEmo   | 0.601 | 0.713 |
| DATN-1    | 0.582 | 0.543 |
| My model  | 0.659 | 0.479 |

## References
**SpanEmo Model** \
SpanEmo: Casting Multi-label Emotion Classification as Span-prediction (Alhuzali & Ananiadou, EACL 2021), \
https://aclanthology.org/2021.eacl-main.135.pdf \
**DATN-1 Model** \
Improving Multi-label Emotion Classification via Sentiment Classification with Dual Attention Transfer Network (Yu et al., EMNLP 2018), https://aclanthology.org/D18-1137.pdf
