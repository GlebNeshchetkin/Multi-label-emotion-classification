import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score

def evaluation_sentiment(modelSm, test_dataloader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    modelSm.eval()
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for batch in tqdm(test_dataloader, desc="Validation"):
            input_ids, attention_mask, labels = batch
            input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)

            logits = modelSm(input_ids=input_ids, attention_mask=attention_mask)
            _, preds = torch.max(logits, dim=1)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    
    return all_labels, all_preds

def evaluation_emotion(modelSm, modelEm, test_dataloader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    modelEm.eval()
    all_labels, all_preds = [], []

    with torch.no_grad():
        for batch in test_dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            logitsSm = modelSm(input_ids=input_ids, attention_mask=attention_mask)
            last_layer_sm = modelSm.get_last_hidden_layer()

            logits = modelEm(input_ids, attention_mask, last_layer_sm)
            preds = torch.sigmoid(logits)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    all_preds_binary = (np.array(all_preds) > 0.5).astype(int)
    return all_labels, all_preds_binary

def calculate_accuracy(targets, predictions):
    number_of_all_predictions = 0
    number_of_correct_predictions = 0

    for pred, target in zip(predictions, targets):
      for p,t in zip(pred, target):
        number_of_all_predictions += 1
        if p==t:
          number_of_correct_predictions += 1

    accuracy = number_of_correct_predictions / number_of_all_predictions
    return accuracy

def get_metrics_sm(targets, predictions):
    accuracy = accuracy_score(targets, predictions)
    f1_score_ = f1_score(targets, predictions, average='micro')
    return accuracy, f1_score_

def get_metrics_em(targets, predictions):
    accuracy = calculate_accuracy(targets=targets, predictions=predictions)
    f1_score_ = f1_score(targets, predictions, average='micro')
    return accuracy, f1_score_