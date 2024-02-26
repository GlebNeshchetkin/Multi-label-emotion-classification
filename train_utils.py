import torch
from tqdm import tqdm
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR
import torch.nn as nn
import numpy as np
from evaluation import get_metrics_em

# Training for Sentiment Classification

def train_sentiment_model(modelSm, train_dataloader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(modelSm.parameters(), lr=0.0001)
    scheduler = MultiStepLR(optimizer, milestones=[10, 20, 30], gamma=0.5)
    
    # Training loop
    num_epochs = 40
    for epoch in range(num_epochs):
        modelSm.train()
        total_loss = 0.0
        num_batches = 0

        for batch in tqdm(train_dataloader, desc=f"Epoch {epoch}"):
            input_ids, attention_mask, labels = batch
            input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)

            optimizer.zero_grad()
            logits = modelSm(input_ids=input_ids, attention_mask=attention_mask)

            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            num_batches += 1

        average_loss = total_loss / num_batches
        print(f"Epoch {epoch + 1}/{num_epochs}, Average Loss: {average_loss:.4f}")
    
    return modelSm
    
    
# Training for Emotion Classification

def train_emotion_model(modelSm, modelEm, train_dataloader, val_dataloader, class_weights):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = nn.MultiLabelSoftMarginLoss(weight=class_weights)
    optimizer = Adam(modelEm.parameters(), lr=0.0005)
    scheduler = MultiStepLR(optimizer, milestones=[3,5, 7, 8], gamma=0.1)
    
    # Training loop
    num_epochs = 10
    for epoch in range(num_epochs):
        total_loss = 0.0
        modelEm.train()
        
        for batch in tqdm(train_dataloader, desc=f"Epoch {epoch}"):
            
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            logitsSm = modelSm(input_ids=input_ids, attention_mask=attention_mask)
            last_layer_sm = modelSm.get_last_hidden_layer()

            optimizer.zero_grad()
            logits = modelEm(input_ids, attention_mask,last_layer_sm)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()
            
        # Validation (optional)
        modelEm.eval()
        all_labels, all_preds = [], []
        with torch.no_grad():
            for batch in val_dataloader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['label'].to(device)

                logitsSm = modelSm(input_ids=input_ids, attention_mask=attention_mask)
                last_layer_sm = modelSm.get_last_hidden_layer()
                
                logits = modelEm(input_ids, attention_mask, last_layer_sm)
                preds = torch.sigmoid(logits)

                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())
        
        all_preds_binary = (np.array(all_preds) >= 0.5).astype(int)
        acc_em, f1_em = get_metrics_em(targets=all_labels,predictions=all_preds_binary)
        print(f"Emotion Classification\nAccuracy Score: {acc_em:.3f}, F1 Score: {f1_em:.3f}")

        average_loss = total_loss / len(train_dataloader)
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {average_loss}")
        
    return modelEm