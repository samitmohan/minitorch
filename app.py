import streamlit as st
import numpy as np
from minitorch.tensor import Tensor, no_grad
from minitorch.module import Sequential
from minitorch.layers import Linear, ReLU
from minitorch.loss import mse_loss, cross_entropy_loss
from minitorch.optim import SGD, Adam
from minitorch.grad_check import numerical_gradient

st.set_page_config(page_title="MiniTorch Playground", layout="wide")
st.title("MiniTorch Playground")

tab1, tab2, tab3 = st.tabs(["Autograd Playground", "Training", "Gradient Checker"])

# Tab 1: Autograd Playground
with tab1:
    st.header("Autograd Playground")
    st.write("Enter an expression to compute and see its gradient.")

    col1, col2 = st.columns(2)
    with col1:
        x_val = st.number_input("Value of x:", value=2.0, step=0.1, key="autograd_x")
        expr = st.selectbox("Expression:", [
            "y = x^2 + 3x",
            "y = sin(x) via Taylor",
            "y = exp(x)",
            "y = 1/x",
        ])

    x = Tensor(np.array(x_val), requires_grad=True)

    if expr == "y = x^2 + 3x":
        y = x ** 2 + x * 3.0
        formula = "dy/dx = 2x + 3"
    elif expr == "y = sin(x) via Taylor":
        # Taylor: x - x^3/6 + x^5/120
        y = x - (x ** 3) / 6.0 + (x ** 5) / 120.0
        formula = "dy/dx = 1 - x^2/2 + x^4/24 (Taylor approx)"
    elif expr == "y = exp(x)":
        y = x.exp()
        formula = "dy/dx = exp(x)"
    else:
        y = 1.0 / x
        formula = "dy/dx = -1/x^2"

    y.backward()

    with col2:
        st.metric("y value", f"{float(y.data):.6f}")
        st.metric("dy/dx", f"{float(x.grad):.6f}")
        st.caption(formula)

    st.subheader("Computation Graph")
    nodes = []
    visited = set()
    def collect(v):
        if id(v) not in visited:
            visited.add(id(v))
            for child in v._prev:
                collect(child)
            if v._op:
                nodes.append(v._op)
    collect(y)
    if nodes:
        st.code(" -> ".join(nodes), language=None)
    else:
        st.write("(leaf node)")

# Tab 2: Training
with tab2:
    st.header("Training")

    with st.sidebar:
        st.subheader("Hyperparameters")
        task = st.selectbox("Task:", ["Linear Regression", "MNIST MLP"])
        optimizer_name = st.selectbox("Optimizer:", ["SGD", "Adam"])
        lr = st.slider("Learning rate:", 0.0001, 0.5, 0.01, format="%.4f")
        epochs = st.slider("Epochs:", 5, 100, 20)

        if task == "MNIST MLP":
            n_train = st.slider("Training samples:", 500, 5000, 1000, step=500)
            batch_size = st.slider("Batch size:", 32, 256, 100, step=32)

    if task == "Linear Regression":
        if st.button("Train Linear Model", key="train_linear"):
            np.random.seed(0)
            x_np = np.random.rand(100, 1).astype(np.float32)
            y_np = 2 * x_np + 1 + 0.1 * np.random.randn(100, 1).astype(np.float32)

            x_t = Tensor(x_np, requires_grad=False)
            y_t = Tensor(y_np, requires_grad=False)

            model = Linear(1, 1)
            params = model.parameters()
            opt = SGD(params, lr=lr) if optimizer_name == "SGD" else Adam(params, lr=lr)

            loss_history = []
            progress = st.progress(0)
            chart_placeholder = st.empty()

            for epoch in range(epochs):
                pred = model(x_t)
                loss = mse_loss(pred, y_t)
                opt.zero_grad()
                loss.backward()
                opt.step()
                loss_history.append(float(loss.data))
                progress.progress((epoch + 1) / epochs)
                chart_placeholder.line_chart(loss_history, y_label="Loss", x_label="Epoch")

            st.success(f"Done. Final loss: {loss_history[-1]:.6f}")
            c1, c2 = st.columns(2)
            c1.metric("Learned weight", f"{model.weight.data.flatten()[0]:.4f}", delta="target: 2.0")
            c2.metric("Learned bias", f"{model.bias.data.flatten()[0]:.4f}", delta="target: 1.0")

    else:
        if st.button("Train MNIST MLP", key="train_mnist"):
            from sklearn.datasets import fetch_openml
            from sklearn.model_selection import train_test_split
            from minitorch.data import DataLoader

            status = st.status("Loading MNIST...")
            with status:
                mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='liac-arff')
                X = mnist.data.astype(np.float32) / 255.0
                y = mnist.target.astype(np.int64)
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                X_train = X_train[:n_train]
                y_train = y_train[:n_train]
                X_test = X_test[:500]
                y_test = y_test[:500]

                def one_hot(labels, nc=10):
                    oh = np.zeros((labels.shape[0], nc), dtype=np.float32)
                    oh[np.arange(labels.shape[0]), labels] = 1.0
                    return oh

                y_train_oh = one_hot(y_train)
                y_test_oh = one_hot(y_test)
                st.write(f"Loaded {n_train} train / 500 test samples")

            model = Sequential(Linear(784, 128), ReLU(), Linear(128, 10))
            params = model.parameters()
            opt = SGD(params, lr=lr, momentum=0.9) if optimizer_name == "SGD" else Adam(params, lr=lr)
            loader = DataLoader(X_train, y_train_oh, batch_size=batch_size, shuffle=True)

            loss_history = []
            progress = st.progress(0)
            chart_placeholder = st.empty()
            metrics_placeholder = st.empty()

            model.train()
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
                progress.progress((epoch + 1) / epochs)
                chart_placeholder.line_chart(loss_history, y_label="Loss", x_label="Epoch")

            # evaluation
            model.eval()
            with no_grad():
                X_test_t = Tensor(X_test, requires_grad=False)
                logits_test = model(X_test_t)
                preds = np.argmax(logits_test.data, axis=1)
            accuracy = np.mean(preds == y_test)

            st.success(f"Test Accuracy: {accuracy * 100:.2f}%")

            # sample predictions grid
            st.subheader("Sample Predictions")
            cols = st.columns(10)
            for i in range(10):
                with cols[i]:
                    img = X_test[i].reshape(28, 28)
                    st.image(img, caption=f"Pred: {preds[i]}\nTrue: {y_test[i]}", width=80)

# Tab 3: Gradient Checker
with tab3:
    st.header("Gradient Checker")
    st.write("Compare analytic vs numerical gradients for each operation.")

    op = st.selectbox("Operation:", [
        "add", "sub", "mul", "div", "pow", "exp", "log",
        "matmul", "sum", "mean",
    ])

    if st.button("Check Gradient", key="check_grad"):
        np.random.seed(42)
        a = Tensor(np.random.randn(3, 4).astype(np.float32), requires_grad=True)
        b = Tensor((np.abs(np.random.randn(3, 4)) + 0.5).astype(np.float32), requires_grad=True)

        ops = {
            "add": lambda: (a + b).sum(),
            "sub": lambda: (a - b).sum(),
            "mul": lambda: (a * b).sum(),
            "div": lambda: (a / b).sum(),
            "pow": lambda: (Tensor(np.abs(a.data) + 0.1, requires_grad=True) ** 2).sum(),
            "exp": lambda: (a * 0.5).exp().sum(),
            "log": lambda: Tensor(np.abs(a.data) + 0.1, requires_grad=True).log().sum(),
            "matmul": lambda: (Tensor(np.random.randn(3, 4).astype(np.float32), requires_grad=True) @ Tensor(np.random.randn(4, 5).astype(np.float32), requires_grad=True)).sum(),
            "sum": lambda: a.sum(),
            "mean": lambda: a.mean(),
        }

        f = ops[op]
        a.zero_grad()
        b.zero_grad()
        loss = f()
        loss.backward()

        analytic_a = a.grad.copy() if a.grad is not None else np.zeros_like(a.data)
        numerical_a = numerical_gradient(f, [a])[0]

        col1, col2, col3 = st.columns(3)
        with col1:
            st.subheader("Analytic")
            st.dataframe(analytic_a, use_container_width=True)
        with col2:
            st.subheader("Numerical")
            st.dataframe(numerical_a, use_container_width=True)
        with col3:
            st.subheader("Abs Difference")
            diff = np.abs(analytic_a - numerical_a)
            st.dataframe(diff, use_container_width=True)

        max_diff = diff.max()
        if max_diff < 1e-3:
            st.success(f"Gradient check passed. Max diff: {max_diff:.2e}")
        else:
            st.error(f"Gradient check failed. Max diff: {max_diff:.2e}")
