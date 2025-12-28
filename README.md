Next-word-Prediction-NLP
📌 Project Overview

This project implements a Next Word Prediction system using Natural Language Processing (NLP) and Bidirectional LSTM (BiLSTM).
The model predicts the most likely next word based on a given sequence of input words.

Next-word prediction is widely used in:

Search engines

Chatbots

Text auto-completion

Language modeling applications

🧠 Model Architecture

The model is built using TensorFlow/Keras with the following architecture:

Input Layer (15 words)
↓
Embedding Layer (100 dimensions)
↓
Bidirectional LSTM
↓
Dropout
↓
Bidirectional LSTM
↓
Dropout
↓
Dense Layer (Softmax Output)

📊 Model Summary
Layer	Output Shape	Parameters
Embedding	(None, 15, 100)	275,100
BiLSTM	(None, 15, 300)	301,200
BiLSTM	(None, 300)	541,200
Dense	(None, 2751)	828,051
Total Parameters		1,945,551
📂 Dataset

Text-based dataset

Tokenized and padded to a fixed length of 15 words

Vocabulary size: 2,751 words

Converted into input–output word sequences for training



🧪 Sample Prediction

Input:

"Machine learning is"


Predicted Output:

"the"
