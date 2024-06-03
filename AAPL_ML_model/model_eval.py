import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# Define the model class (same as the one used during training)
class SimpleLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(1, x.size(0), hidden_size).to(x.device)
        c0 = torch.zeros(1, x.size(0), hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

# Hyperparameters (ensure they match the ones used during training)
input_size = 7  # Number of features
hidden_size = 50
output_size = 1
window_size = 60  # Ensure this matches the window size used during training

# Load the saved model
model = SimpleLSTM(input_size, hidden_size, output_size)
model.load_state_dict(torch.load('optimized_model.pth'))
model.eval()  # Set the model to evaluation mode

# Load the test data
X_test = np.load('X_test.npy')
y_test = np.load('y_test.npy')

# Convert test data to PyTorch tensors
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

# Evaluate the model
criterion = nn.MSELoss()
model.eval()
with torch.no_grad():
    test_predictions = model(X_test_tensor)
    test_loss = criterion(test_predictions, y_test_tensor)
    print(f'Test Loss: {test_loss.item():.4f}')

# Plot the results
test_predictions_np = test_predictions.numpy().flatten()
y_test_np = y_test_tensor.numpy().flatten()

plt.figure(figsize=(12, 6))
plt.plot(y_test_np, label='Actual Close Prices')
plt.plot(test_predictions_np, label='Predicted Close Prices')
plt.title('Actual vs Predicted Close Prices')
plt.xlabel('Time')
plt.ylabel('Close Price')
plt.legend()
plt.savefig('model_accuracy_plot.png')  # Save the plot
plt.show()
