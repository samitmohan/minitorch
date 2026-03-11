import numpy as np
from minitorch.tensor import Tensor, no_grad
from minitorch.module import Sequential
from minitorch.layers import Linear, ReLU, Dropout
from minitorch.conv import Conv2d, MaxPool2d, Flatten
from minitorch.loss import cross_entropy_loss
from minitorch.optim import Adam
from minitorch.data import DataLoader

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split


def one_hot(labels, num_classes=10):
    oh = np.zeros((labels.shape[0], num_classes), dtype=np.float32)
    oh[np.arange(labels.shape[0]), labels] = 1.0
    return oh


def build_mlp():
    return Sequential(
        Linear(784, 128),
        ReLU(),
        Linear(128, 10),
    )


def build_cnn():
    return Sequential(
        Conv2d(1, 16, 3, padding=1),   # 28x28 -> 28x28
        ReLU(),
        MaxPool2d(2),                    # 28x28 -> 14x14
        Conv2d(16, 32, 3, padding=1),   # 14x14 -> 14x14
        ReLU(),
        MaxPool2d(2),                    # 14x14 -> 7x7
        Flatten(),                       # 32*7*7 = 1568
        Linear(1568, 128),
        ReLU(),
        Dropout(0.25),
        Linear(128, 10),
    )


def run_mnist_example(model_type="mlp", n_train=10000, n_test=2000, epochs=15, batch_size=100, lr=0.001):
    print("Fetching MNIST data...")
    mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='liac-arff')
    X = mnist.data.astype(np.float32) / 255.0
    y = mnist.target.astype(np.int64)

    X_train_all, X_test_all, y_train_all, y_test_all = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    X_train = X_train_all[:n_train]
    y_train = y_train_all[:n_train]
    X_test = X_test_all[:n_test]
    y_test = y_test_all[:n_test]

    y_train_oh = one_hot(y_train, 10)
    y_test_oh = one_hot(y_test, 10)

    if model_type == "cnn":
        X_train = X_train.reshape(-1, 1, 28, 28)
        X_test = X_test.reshape(-1, 1, 28, 28)
        model = build_cnn()
        print("Model: CNN")
    else:
        model = build_mlp()
        print("Model: MLP")

    params = model.parameters()
    opt = Adam(params, lr=lr)
    loader = DataLoader(X_train, y_train_oh, batch_size=batch_size, shuffle=True)

    loss_history = []
    model.train()

    print("Training...")
    for epoch in range(epochs):
        epoch_loss = 0.0
        n_batches = 0
        for xb_np, yb_np in loader:
            xb = Tensor(xb_np, requires_grad=False)
            yb = Tensor(yb_np, requires_grad=False)

            logits = model(xb)
            loss = cross_entropy_loss(logits, yb)

            opt.zero_grad()
            loss.backward()
            opt.step()

            epoch_loss += float(loss.data)
            n_batches += 1

        avg_loss = epoch_loss / n_batches
        loss_history.append(avg_loss)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

    # evaluation
    model.eval()
    test_loader = DataLoader(X_test, y_test_oh, batch_size=200, shuffle=False)
    correct = 0
    total = 0
    with no_grad():
        for xb_np, yb_np in test_loader:
            xb = Tensor(xb_np, requires_grad=False)
            logits = model(xb)
            preds = np.argmax(logits.data, axis=1)
            labels = np.argmax(yb_np, axis=1)
            correct += np.sum(preds == labels)
            total += len(labels)

    accuracy = correct / total
    print(f"Test Accuracy: {accuracy * 100:.2f}%")

    return loss_history, accuracy


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["mlp", "cnn"], default="mlp")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--n-train", type=int, default=10000)
    parser.add_argument("--n-test", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.001)
    args = parser.parse_args()

    run_mnist_example(
        model_type=args.model,
        n_train=args.n_train,
        n_test=args.n_test,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
    )
