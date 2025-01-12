import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack
from sklearn.metrics import mean_absolute_error, accuracy_score

# 擴展模擬保險爭議案例數據
train_data = pd.DataFrame({
    'dispute_type': [
        'Delayed Payment', 'Policy Exclusion', 'Claim Rejection', 
        'Delayed Payment', 'Claim Rejection', 'Policy Exclusion',
        'Claim Rejection', 'Delayed Payment', 'Policy Exclusion',
        'Delayed Payment', 'Claim Rejection', 'Policy Exclusion'
    ],
    'claim_amount': [5000, 10000, 15000, 2000, 12000, 7000, 9000, 3000, 11000, 4500, 13000, 6000],
    'dispute_description': [
        'Payment delayed due to internal errors', 
        'Policy excludes certain damages', 
        'Claim rejected for missing documents', 
        'Payment pending for over a month', 
        'Documents not provided on time', 
        'Exclusions not clearly communicated',
        'Claim denied without reason', 
        'Payment delay due to investigation', 
        'Policy terms exclude this case', 
        'Payment overdue, no updates', 
        'Rejected claim due to missing receipts', 
        'Damages excluded from policy terms'
    ],
    'claim_resolution': [1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0]  # 1: 已解決, 0: 未解決
})

# 額外生成需要被預測的數據
prediction_data = pd.DataFrame({
    'dispute_type': [
        'Policy Exclusion', 'Delayed Payment', 'Claim Rejection'
    ],
    'claim_amount': [8000, 4000, 14000],
    'dispute_description': [
        'Policy does not cover third-party damages',
        'Payment delayed for unclear reasons',
        'Claim rejected for insufficient proof'
    ]
})

# 文本處理：使用 TfidfVectorizer 將爭議描述轉換為數字
vectorizer = TfidfVectorizer()
X_text = vectorizer.fit_transform(train_data['dispute_description'])
X_text_pred = vectorizer.transform(prediction_data['dispute_description'])

# 數值特徵處理：對 claim_amount 進行標準化
scaler = StandardScaler()
X_claim_amount = scaler.fit_transform(train_data[['claim_amount']])
X_claim_amount_pred = scaler.transform(prediction_data[['claim_amount']])

# 類別特徵處理：將 dispute_type 轉換為數字類型
encoder = LabelEncoder()
X_dispute_type = encoder.fit_transform(train_data['dispute_type']).reshape(-1, 1)
X_dispute_type_pred = encoder.transform(prediction_data['dispute_type']).reshape(-1, 1)

# 合併所有特徵
X = hstack([X_claim_amount, X_dispute_type, X_text]).toarray()
X_pred = hstack([X_claim_amount_pred, X_dispute_type_pred, X_text_pred]).toarray()

# 目標變量：claim_resolution
y = train_data['claim_resolution'].values

# 分割數據集為訓練集和測試集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 將數據調整為三維格式 (batch_size, seq_len, input_dim)
X_train_tensor = torch.tensor(X_train, dtype=torch.float32).unsqueeze(1)  # 增加一個維度作為時間步
X_test_tensor = torch.tensor(X_test, dtype=torch.float32).unsqueeze(1)
X_pred_tensor = torch.tensor(X_pred, dtype=torch.float32).unsqueeze(1)

y_train_tensor = torch.tensor(y_train, dtype=torch.long)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

# LSTM模型
class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        out = self.fc(lstm_out[:, -1, :])  # 只取最後一個時間步的輸出
        return out

# XLSTM模型 (加入矩陣儲存特性)
class XLSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(XLSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.matrix_transform = nn.Linear(hidden_dim, hidden_dim * 2)  # 增加矩陣特徵層
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        matrix_features = self.matrix_transform(lstm_out[:, -1, :])
        out = self.fc(matrix_features)
        return out

# Transformer模型
class TransformerModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(TransformerModel, self).__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=input_dim, nhead=1)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=1)
        self.fc = nn.Linear(input_dim, output_dim)
    
    def forward(self, x):
        x = x.permute(1, 0, 2)  # Transformer需要 (seq_len, batch_size, input_dim)
        transformer_out = self.transformer_encoder(x)
        out = self.fc(transformer_out[-1, :, :])  # 只取最後一個時間步
        return out

# 訓練與評估函數
def train_and_evaluate(model, X_train_tensor, X_test_tensor, y_train_tensor, y_test_tensor, epochs=10, lr=0.001):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        loss.backward()
        optimizer.step()
        
        if (epoch+1) % 2 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')
    
    model.eval()
    with torch.no_grad():
        outputs = model(X_test_tensor)
        _, predicted = torch.max(outputs, 1)
        mae = mean_absolute_error(y_test_tensor, predicted)
        accuracy = accuracy_score(y_test_tensor, predicted)
        print(f'MAE: {mae:.4f}, Accuracy: {accuracy:.4f}')
        return mae, accuracy

# 預測函數
def predict(model, X_pred_tensor):
    model.eval()
    with torch.no_grad():
        outputs = model(X_pred_tensor)
        _, predicted = torch.max(outputs, 1)
        return predicted.numpy()

# 訓練並評估 LSTM 模型
input_dim = X_train.shape[1]  # 輸入維度
hidden_dim = 64  # 隱藏層維度
output_dim = 2  # 類別數

print("Training LSTM model...")
lstm_model = LSTMModel(input_dim, hidden_dim, output_dim)
train_and_evaluate(lstm_model, X_train_tensor, X_test_tensor, y_train_tensor, y_test_tensor)

# 訓練並評估 XLSTM 模型
print("Training XLSTM model...")
xlstm_model = XLSTMModel(input_dim, hidden_dim, output_dim)
train_and_evaluate(xlstm_model, X_train_tensor, X_test_tensor, y_train_tensor, y_test_tensor)

# 訓練並評估 Transformer 模型
print("Training Transformer model...")
transformer_model = TransformerModel(input_dim, hidden_dim, output_dim)
train_and_evaluate(transformer_model, X_train_tensor, X_test_tensor, y_train_tensor, y_test_tensor)

# 預測需要被預測的數據
print("Predicting with XLSTM model...")
predictions = predict(xlstm_model, X_pred_tensor)
print("Predictions for new data:", predictions)
