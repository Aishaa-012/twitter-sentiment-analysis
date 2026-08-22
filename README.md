# Twitter Sentiment Analysis (DistilBERT)

Fine-tuned DistilBERT model for 3-class sentiment analysis (positive/negative/neutral) on tweet data.

## 🔗 Live Demo
Try it here: [Sentiment Analysis Demo](https://twitter-sentiment-analysis-llynpjmbnesfeuumiickjf.streamlit.app/)

## 🛠️ Tech Stack
- **Model:** DistilBERT (fine-tuned)
- **Framework:** PyTorch, Hugging Face Transformers
- **Deployment:** Streamlit, Hugging Face Hub
- **Language:** Python
- **Data Processing:** Pandas, Scikit-learn

## Results
- **Test Accuracy: 79.09%**
- Precision/Recall/F1 by class:
  - Negative: 0.78 / 0.81 / 0.79
  - Neutral: 0.77 / 0.74 / 0.76
  - Positive: 0.84 / 0.83 / 0.83

## What I Learned / Fixed
While training, I noticed validation accuracy peaked after epoch 1 (79.11%) and then declined over subsequent epochs, even though training loss kept dropping — a clear sign of overfitting. I fixed this by adding checkpointing that saves the model only when validation accuracy improves, and evaluating on that best checkpoint instead of just the final epoch.

## Model
- Base model: `distilbert-base-uncased`
- Fine-tuned for 3 epochs, batch size 16, learning rate 2e-5
- Framework: PyTorch + HuggingFace Transformers

## Dataset
Dataset was provided by my instructor for coursework. To run this notebook, you'll need your own copy of `train.csv` and `test.csv` with `text` and `sentiment` columns, mounted via Google Drive.

## How to Run
1. Open the notebook in Google Colab
2. Mount your Google Drive with the dataset
3. Update the dataset path in the data-loading cell
4. Run all cells (Runtime → Run all)
