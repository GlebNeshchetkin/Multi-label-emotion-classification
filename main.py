import torch
import pandas as pd
from data_preprocessing import preprocess_sentiment_classification_train, preprocess_emotion_classification, preprocess_emotion_classification, preprocess_sentiment_classification_test
from model_defenitions import BertClassifier, BertClassifierEm
from train_utils import train_sentiment_model, train_emotion_model
from evaluation import evaluation_sentiment, evaluation_emotion, get_metrics_sm, get_metrics_em
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split

def main():
    
    # Training
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    data_train = pd.read_csv('trainingdata-all-annotations.csv')
    train_dataloader, val_dataloader = preprocess_sentiment_classification_train(data_train_in=data_train)
    modelSm = BertClassifier(num_labels=3, freeze_bert=False)
    modelSm.to(device)
    modelSm = train_sentiment_model(modelSm=modelSm, train_dataloader=train_dataloader)
    torch.save(modelSm.state_dict(), 'modelSm_cpu_1.0_params.pth')
    
    data_train_em = pd.read_csv('2018-E-c-En-train.csv')
    data_test_em = pd.read_csv('2018-E-c-En-test-gold.csv')
    
    labels_em = data_train_em.drop(columns=['ID','Tweet'])
    labels_em = torch.tensor(labels_em.values, dtype=torch.float32)
    class_weights = 1.0 / (labels_em.sum(dim=0)).to(device)
    class_weights = class_weights / class_weights.sum()
    
    train_dataloader_em = preprocess_emotion_classification(data=data_train_em)
    test_dataloader_em = preprocess_emotion_classification(data=data_test_em)
    num_labels = len(data_train_em.columns) - 2  # Assuming the first two columns are 'ID' and 'Tweet'
    
    test_indices, val_indices = train_test_split(range(len(test_dataloader_em.dataset)), test_size=0.2, random_state=42)
    
    val_subset = Subset(test_dataloader_em.dataset, val_indices)
    val_dataloader = DataLoader(val_subset, batch_size=test_dataloader_em.batch_size, shuffle=False)
    
    modelEm = BertClassifierEm(num_labels=num_labels, freeze_bert=False)
    modelEm.to(device)
    modelEm = train_emotion_model(modelSm=modelSm, modelEm=modelEm,train_dataloader=train_dataloader_em, val_dataloader=val_dataloader, class_weights=class_weights)
    torch.save(modelEm.state_dict(), 'modelEm_cpu_1.0_params.pth')
    
    # Evaluation
    
    data_test_sm = pd.read_csv('trialdata-all-annotations.csv')
    data_test_em = pd.read_csv('2018-E-c-En-test-gold.csv')
    
    test_dataloader_sm = preprocess_sentiment_classification_test(data_test_in=data_test_sm)
    test_dataloader_em = preprocess_emotion_classification(data=data_test_em)
    
    all_labels_sm, all_preds_sm = evaluation_sentiment(modelSm=modelSm, test_dataloader=test_dataloader_sm)
    all_labels_em, all_preds_em = evaluation_emotion(modelSm=modelSm, modelEm=modelEm, test_dataloader=test_dataloader_em)
    
    acc_sm, f1_sm = get_metrics_sm(targets=all_labels_sm,predictions=all_preds_sm)
    acc_em, f1_em = get_metrics_em(targets=all_labels_em,predictions=all_preds_em)
    
    print(f"Sentiment Classification\nAccuracy Score: {acc_sm:.3f}, F1 Score: {f1_sm:.3f}\nEmotion Classification\nAccuracy Score: {acc_em:.3f}, F1 Score: {f1_em:.3f}")
    
if __name__ == "__main__":
    main()