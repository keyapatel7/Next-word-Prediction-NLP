# Next-word-Prediction-NLP
Next Word Prediction Using BiLSTM
📌 Project Overview

This project implements a Next Word Prediction system using Deep Learning (Bidirectional LSTM).
The model predicts the most probable next word based on a given sequence of words.

Such models are commonly used in:

Search engines

Chatbots

Text auto-completion

NLP-based applications

🧠 Model Architecture

The model is built using TensorFlow/Keras and follows the architecture below:

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

Tokenized and padded to sequence length of 15

Vocabulary size: 2,751 words

Converted into input–output word sequences for training

⚙️ Technologies Used

Python

TensorFlow / Keras

NumPy

Natural Language Processing (NLP)

LSTM / BiLSTM

Jupyter Notebook / Google Colab

🚀 Training Details

Loss Function: Categorical Crossentropy

Optimizer: Adam

Epochs: 20

Batch Size: 64

📈 Training Performance
Metric	Value
Training Accuracy	~24%
Validation Accuracy	~12%
Final Training Loss	3.58
Final Validation Loss	6.75

Note: Lower accuracy is expected due to the large vocabulary size and complexity of multi-class word prediction.

🧪 Example Prediction

Input:

"Deep learning is"


Predicted Output:

"the"

📁 Project Structure
├── data/
│   └── text_data.txt
├── model/
│   └── next_word_model.h5
├── notebook/
│   └── next_word_prediction.ipynb
├── README.md
└── requirements.txt

▶️ How to Run the Project
1️⃣ Install Required Libraries
pip install -r requirements.txt

2️⃣ Run the Notebook
jupyter notebook

3️⃣ Train the Model

Run all cells in:

next_word_prediction.ipynb

4️⃣ Predict Next Word
predict_next_word("Machine learning is")

🚀 Future Enhancements

Use pre-trained embeddings (GloVe / Word2Vec)

Add Attention Mechanism

Improve accuracy with hyperparameter tuning

Implement Transformer-based architecture

Deploy using Flask or Streamlit
