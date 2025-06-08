import numpy as np
from minitorch.tensor import Tensor
from minitorch.layers import Linear, ReLU
from minitorch.loss import cross_entropy_loss
from minitorch.optim import SGD, Adam

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

print("Fetching MNIST data...")
mnist = fetch_openml('mnist_784', version=1)
X = mnist.data.astype(np.float32) / 255.0
y = mnist.target.astype(np.int64)

X_train_all, X_test_all, y_train_all, y_test_all = train_test_split(
    X, y, test_size=0.2, random_state=42
)
X_train = X_train_all[:1000]
y_train = y_train_all[:1000]
X_test = X_test_all[:200]
y_test = y_test_all[:200]

def one_hot(labels, num_classes=10):
    oh = np.zeros((labels.shape[0], num_classes), dtype=np.float32)
    oh[np.arange(labels.shape[0]), labels] = 1.0
    return oh

y_train_oh = one_hot(y_train, 10)
y_test_oh = one_hot(y_test, 10)

X_t = Tensor(X_train, requires_grad=False)
y_t = Tensor(y_train_oh, requires_grad=False)

layer1 = Linear(784, 128)
relu = ReLU()
layer2 = Linear(128, 10)

params = layer1.parameters() + layer2.parameters()
# opt = SGD(params, lr=0.01, momentum=0.9)
opt = Adam(params, lr=1e-2)


epochs = 10
batch_size = 100
num_batches = X_train.shape[0] // batch_size

print("Training")
for epoch in range(epochs):
    epoch_loss = 0.0
    for i in range(num_batches):
        start = i * batch_size
        end = start + batch_size
        xb = Tensor(X_train[start:end], requires_grad=False)
        yb = Tensor(y_train_oh[start:end], requires_grad=False)

        out1 = layer1(xb)
        act1 = relu(out1)
        logits = layer2(act1)
        loss = cross_entropy_loss(logits, yb)

        opt.zero_grad()
        loss.backward()
        opt.step()

        epoch_loss += loss.data

    avg_loss = epoch_loss / num_batches
    print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

X_test_t = Tensor(X_test, requires_grad=False)
out1_test = layer1(X_test_t)
act1_test = relu(out1_test)
logits_test = layer2(act1_test)
preds = np.argmax(logits_test.data, axis=1)
accuracy = np.mean(preds == y_test)
print(f"Test Accuracy: {accuracy * 100:.2f}%")