import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load the CSV file
file_path = 'Data/chb01/chbmit_preprocessed_data.csv'  # Replace with the actual path to your CSV file
data = pd.read_csv(file_path)

# Extract features and labels
features = data.iloc[:, :-1].values  # Exclude the 'Outcome' column
labels = data['Outcome'].values

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Standardize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Convert NumPy arrays to PyTorch tensors
X_train_tensor = torch.FloatTensor(X_train).cuda()
y_train_tensor = torch.LongTensor(y_train).cuda()
X_test_tensor = torch.FloatTensor(X_test).cuda()
y_test_tensor = torch.LongTensor(y_test).cuda()

# Add a dimension for the sequence
X_train_tensor = X_train_tensor.unsqueeze(1)
X_test_tensor = X_test_tensor.unsqueeze(1)

# Define a simple recurrent neural network architecture
class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleRNN, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        rnn_out, _ = self.rnn(x)
        output = self.fc(rnn_out[:, -1, :])  # Use the output from the last time step
        return output

# Instantiate the model and move it to CUDA
input_size = X_train_tensor.size(2)  # Adjusted for the shape of the input data
hidden_size = 64  # You can adjust this based on your requirements
output_size = 2
model = SimpleRNN(input_size, hidden_size, output_size).cuda()

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Create a DataLoader for batching and shuffling the data
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    for inputs, targets in train_dataloader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

    print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}')

# Evaluate the model on the test set
with torch.no_grad():
    model.eval()
    test_outputs = model(X_test_tensor)
    predicted_labels = torch.argmax(test_outputs, dim=1)
    accuracy = torch.sum(predicted_labels == y_test_tensor).item() / len(y_test)
    print(f'Test Accuracy: {accuracy * 100:.2f}%')
