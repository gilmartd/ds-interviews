import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import numpy as np
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv('dev.csv')

# Select target and features
features = ['wtempc', 'precp_in', 'windspeed_knots', 'relh_pct', 'wvel_fps']
target = 'turb_fnu'

# Handle missing values using mean imputation
imputer = SimpleImputer(strategy='mean')
df[features] = imputer.fit_transform(df[features])
df[target] = imputer.fit_transform(df[[target]])

# Splitting the dataset into training, validation, and test sets
X = df[features]
y = df[target]

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

# Convert data to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val.values, dtype=torch.float32).view(-1, 1)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)

# Create a custom dataset
class TurbidityDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# Create DataLoader
train_dataset = TurbidityDataset(X_train_tensor, y_train_tensor)
val_dataset = TurbidityDataset(X_val_tensor, y_val_tensor)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Define the neural network model
class TurbidityModel(nn.Module):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, output_dim):
        super(TurbidityModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim1)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.relu2 = nn.ReLU()
        self.dropout = nn.Dropout(0.3)  # Add dropout to prevent overfitting
        self.fc3 = nn.Linear(hidden_dim2, output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.dropout(x)
        x = self.fc3(x)
        return x

# Initialize the model, loss function, and optimizer
input_dim = len(features)
hidden_dim1 = 64
hidden_dim2 = 32
output_dim = 1

model = TurbidityModel(input_dim, hidden_dim1, hidden_dim2, output_dim)
loss_fn = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)  # Add weight decay for regularization

# Training loop
epochs = 50
train_losses = []
val_losses = []

for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()  # Zero the parameter gradients
        y_pred = model(X_batch)
        loss = loss_fn(y_pred, y_batch)
        loss.backward()  # Backpropagation
        optimizer.step()  # Update parameters
        running_loss += loss.item()

    # Validation step
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            y_pred = model(X_batch)
            loss = loss_fn(y_pred, y_batch)
            val_loss += loss.item()

    # Average loss for this epoch
    train_loss = running_loss / len(train_loader)
    val_loss /= len(val_loader)
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    print(f"Epoch [{epoch+1}/{epochs}], Training Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")

# Plot training and validation loss
plt.figure(figsize=(10, 6))
plt.plot(range(1, epochs + 1), train_losses, label='Training Loss')
plt.plot(range(1, epochs + 1), val_losses, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss Over Epochs')
plt.legend()
plt.grid()
plt.show()

# Evaluation on test data
model.eval()
with torch.no_grad():
    y_test_pred = model(X_test_tensor)
    test_loss = loss_fn(y_test_pred, y_test_tensor).item()
    r2 = r2_score(y_test_tensor.numpy(), y_test_pred.numpy())

print(f"Test Loss (MSE): {test_loss:.4f}, R2 Score: {r2:.4f}")

# Save the trained model
torch.save(model.state_dict(), 'turbidity_model.pth')

# Load the trained model for new data
model.load_state_dict(torch.load('turbidity_model.pth'))
model.eval()  # Set the model to evaluation mode

# Load and preprocess new data
new_data = pd.read_csv('new_data.csv')
new_data[features] = imputer.transform(new_data[features])  # Use the same imputer
new_data_scaled = scaler.transform(new_data[features])  # Use the same scaler

# Convert new data to tensor
new_data_tensor = torch.tensor(new_data_scaled, dtype=torch.float32)

# Make predictions on new data
with torch.no_grad():
    new_predictions = model(new_data_tensor)
print("Predictions on new data:", new_predictions)