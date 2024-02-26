import torch.nn as nn
from transformers import BertModel

class BertClassifier(nn.Module):
    def __init__(self, num_labels, freeze_bert=False):
        super(BertClassifier, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')

        D_in, H, D_out = self.bert.config.hidden_size, 11, num_labels

        self.classifier = nn.Sequential(
            nn.Linear(D_in, H),
            nn.BatchNorm1d(H),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(H, H),
            nn.BatchNorm1d(H),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(H, H),
            nn.BatchNorm1d(H),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(H, H),
            nn.BatchNorm1d(H),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(H, H),
            nn.BatchNorm1d(H),
            nn.ReLU(),
            nn.Linear(H, D_out)
        )

        self.intermediate_representation = None

        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False

    def get_last_hidden_layer(self):
      return self.intermediate_representation

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state_cls = outputs.last_hidden_state[:, 0, :]
        intermediate_representation = self.classifier[:-1](last_hidden_state_cls)
        self.intermediate_representation = intermediate_representation
        logits = self.classifier[-1](intermediate_representation)
        return logits

class BertClassifierEm(nn.Module):
    def __init__(self, num_labels=11, hidden_size=11, num_lstm_layers=2, bidirectional=True):
        
        hidden_size = num_labels
        
        super(BertClassifierEm, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')

        self.lstm = nn.LSTM(
            input_size=self.bert.config.hidden_size,
            hidden_size=hidden_size,
            num_layers=num_lstm_layers,
            batch_first=True,
            dropout=0.1
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.LeakyReLU(),
            nn.Dropout(0.3),

            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.LeakyReLU(),
            nn.Dropout(0.3),

            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.LeakyReLU(),

        )

    def forward(self, input_ids, attention_mask, last_layer_sm):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        lstm_out, _ = self.lstm(pooled_output.unsqueeze(0))
        lstm_out = lstm_out.squeeze(0)
        concatenated_output = lstm_out + last_layer_sm
        probabilities = self.classifier(concatenated_output)
        return probabilities
