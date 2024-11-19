import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class SimpleLinearModel(nn.Module):
    def __init__(self, INPUT_SIZE, OUTPUT_SIZE):
        super(SimpleLinearModel, self).__init__()
        self.linear = nn.Linear(INPUT_SIZE, OUTPUT_SIZE, bias=True)
        # Convert weights and biases to double precision
        self.linear.weight.data = self.linear.weight.data.double()
        self.linear.bias.data = self.linear.bias.data.double()

    def forward(self, x):
        return self.linear(x)

def custom_loss_function(outputs, targets, alpha, w):
    diff = outputs - targets
    conventional_loss_term = alpha * torch.sum(diff ** 2, dim=1)
    weighted_diff = torch.matmul(diff, w)
    prediction_bias_term = (1 - alpha) * weighted_diff ** 2
    loss = conventional_loss_term + prediction_bias_term
    return loss.mean()

class PortfolioOptimizer():
    def __init__(self, linear_model, X, Y):
        self.linear_model = linear_model
        self.X = X
        self.Y = Y
        self.OUTPUT_SIZE = Y.shape[1]
        self.current_training_loss = None
        self.INPUT_SIZE = X.shape[1]
        self.optimizer = optim.SGD(self.linear_model.parameters(), lr=0.01)

    def change_learning_rate(self, new_lr):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr

    def train(self, weights=None, alpha=0.9, n_epoch=100):
        if weights is None:
            weights = [1.0 / self.OUTPUT_SIZE] * self.OUTPUT_SIZE
        w = torch.tensor(weights, dtype=torch.float64)
        w = w.to(self.X.device)

        for _ in range(n_epoch):
            self.linear_model.train()
            outputs = self.linear_model(self.X)
            loss = custom_loss_function(outputs, self.Y, alpha, w)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        self.current_training_loss = loss.item()
        return True

    def get_current_training_loss(self):
        return self.current_training_loss

    def get_training_parameters(self):
        return self.linear_model.linear.weight.data

    def get_training_bias(self):
        return self.linear_model.linear.bias.data

    def eval(self, X_new):
        self.linear_model.eval()
        if isinstance(X_new, np.ndarray):
            X_new = torch.from_numpy(X_new).double()
        X_new = X_new.to(next(self.linear_model.parameters()).device)
        with torch.no_grad():
            predictions = self.linear_model(X_new)
        return predictions.cpu().numpy()

if __name__ == "__main__":
    predictions_by_alpha = {}

    INPUT_SIZE = 5       
    OUTPUT_SIZE = 3      
    N_SAMPLES = 200      
    N_TEST_SAMPLES = 1
    N_EPOCH = int(1E4)
    ALPHA = .1

    np.random.seed(1)
    torch.manual_seed(1)

    X_train = np.random.normal(0, 1, (N_SAMPLES, INPUT_SIZE))

    true_weights = np.random.randn(INPUT_SIZE, OUTPUT_SIZE)
    true_biases = np.random.randn(OUTPUT_SIZE)

    Y_train = X_train.dot(true_weights) + true_biases + np.random.normal(0, 0.5, (N_SAMPLES, OUTPUT_SIZE))

    X_train_tensor = torch.from_numpy(X_train).double()
    Y_train_tensor = torch.from_numpy(Y_train).double()

    X_new = np.random.randn(1, INPUT_SIZE)
    Y_new = X_new.dot(true_weights) + true_biases + np.random.normal(0, 0.5, (1, OUTPUT_SIZE))

    for alpha in range(1, 11):
        model = SimpleLinearModel(INPUT_SIZE, OUTPUT_SIZE).double()
        optimizer = PortfolioOptimizer(model, X_train_tensor, Y_train_tensor)

        optimizer.train(alpha=alpha / 10, n_epoch=N_EPOCH)
        weights = optimizer.get_training_parameters()
        biases = optimizer.get_training_bias()
        predictions_by_alpha[alpha] = optimizer.eval(X_new)

    # Print predictions for each alpha
    for alpha, prediction in predictions_by_alpha.items():
        print(f"Alpha {alpha/10}: Prediction {prediction}")