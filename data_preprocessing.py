import torch
import pandas as pd
from transformers import BertTokenizer
from torch.utils.data import DataLoader, Dataset, TensorDataset, random_split
from imblearn.over_sampling import RandomOverSampler


# Preprocessing for Sentiment Classification

def preprocess_tweets_sentiment(data):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    inputs = tokenizer(data['Tweet'].tolist(), padding=True, truncation=True, return_tensors="pt", max_length=512)

    # Convert string labels to numeric format
    label_dict = {"POSITIVE": 0, "NEITHER": 1, "NEGATIVE": 2}
    labels = torch.tensor([label_dict[label] for label in data['Sentiment'].tolist()], dtype=torch.long)
    return TensorDataset(inputs['input_ids'], inputs['attention_mask'], labels)

def preprocess_sentiment_classification_train(data_train_in):
    columns_to_drop = ['Target', 'Stance', 'Opinion towards']
    data_train = data_train_in.drop(columns=columns_to_drop)
    # data_test = data_test_in.drop(columns=columns_to_drop)
    X = data_train[['ID', 'Tweet']]
    y = data_train['Sentiment']
    
    oversampler = RandomOverSampler(sampling_strategy='not majority', random_state=42)
    X_resampled, y_resampled = oversampler.fit_resample(X, y)
    data_balanced = pd.DataFrame({'ID': X_resampled['ID'], 'Tweet': X_resampled['Tweet'], 'Sentiment': y_resampled})
    data_train = data_balanced

    train_dataset = preprocess_tweets_sentiment(data_train)
    # test_dataset = preprocess_tweets_sentiment(data_test)

    train_size = int(1 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

    batch_size = 50
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    # test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_dataloader, val_dataloader

def preprocess_sentiment_classification_test(data_test_in):
    columns_to_drop = ['Target', 'Stance', 'Opinion towards']
    data_test = data_test_in.drop(columns=columns_to_drop)
    test_dataset = preprocess_tweets_sentiment(data_test)

    batch_size = 50
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return test_dataloader

# Preprocessing for Emotion Classification

class EmotionDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_length):
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        tweet = str(self.data.iloc[idx]['Tweet'])
        label = torch.tensor(self.data.iloc[idx, 2:].astype(float))

        encoding = self.tokenizer(
            tweet,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'label': label
        }

def preprocess_emotion_classification(data):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    max_length = 128

    dataset = EmotionDataset(data, tokenizer, max_length)

    batch_size = 16
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    return dataloader


