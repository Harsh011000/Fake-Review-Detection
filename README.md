# Fake-Review-Detection

## 📌 Overview
This repository contains an LSTM-based deep learning model for Fake-Review-Detection using pre-trained GloVe word embeddings. The project involves:

- **Preprocessing**: Loading GloVe embeddings and preparing an embedding matrix.
- **Model Training**: Implementing an LSTM model with hyperparameter tuning using Ray Tune.
- **Hyperparameter Optimization**: Tuning LSTM layers, units, and learning rate with Bayesian Optimization.
- **Evaluation**: Assessing model performance with accuracy, binary cross-entropy loss, and confusion matrices.

## 🚀 Live Demo  
Try the deployed model here: **[Fake Review Detection App](https://harsh-p-fake-review-detector.hf.space/)**

## 📂 Project Structure
```
📦 LSTM-Text-Classification
├── 📜 Embeddings.ipynb        # Loading GloVe embeddings and preparing embedding matrix
├── 📜 Model Training.ipynb    # Training and evaluating the model
├── 📜 glove.6B.50d.txt        # Pre-trained GloVe embeddings (50d vectors)
├── 📜 README.md               # Project documentation
└── 📜 requirements.txt        # Dependencies and packages required
```


## 🔬 Model Architecture
- **Embedding Layer**: Pre-trained GloVe embeddings (non-trainable)
- **LSTM Layers**: Multiple LSTM layers with tuned units
- **Dense Layer**: Fully connected layer with a sigmoid activation function
- **Optimizer**: Adam with a tuned learning rate
- **Loss Function**: Binary Cross-Entropy

## 📊 Evaluation
The model is evaluated using:
- **Accuracy**
- **Binary Cross-Entropy Loss**
- **Confusion Matrix Visualization**

A sample confusion matrix:
```
                Predicted
            Machine  |  Human
True  Machine    [TN]  |   [FP]
      Human     [FN]  |   [TP]
```

## 🛠️ Hyperparameter Tuning
Hyperparameter optimization is performed using random search. The tuning process explores:
- Number of LSTM layers
- LSTM hidden units
- Learning rate

## 📌 Dependencies
- Python 3.8+
- TensorFlow
- NumPy
- Pandas
- Ray Tune
- Matplotlib
- Seaborn

