import streamlit as st
import numpy as np
from minitorch.tensor import Tensor
from minitorch.layers import Linear, ReLU
from minitorch.loss import mse_loss
from minitorch.optim import SGD, Adam
from mnist_example import run_mnist_example

st.set_page_config(page_title="MiniTorch Playground", layout="centered")

st.title("MiniTorch Demo")
# Tensor operations demo
st.header("Tensor Operations")
x_val = st.number_input("Enter a scalar value for x:", value=2.0)
x = Tensor(x_val, requires_grad=True)
y = x * Tensor(3.0) + Tensor(1.0)
y.backward()
st.write(f"Computed y = 3*x + 1 = {y.data}")
st.write(f"Gradient dy/dx = {x.grad}")

st.markdown("---")
# Linear regression demo
st.header("Train a Linear Model (y = 2x + 1)")
np.random.seed(0)
x_np = np.random.rand(100, 1).astype(np.float32)
y_np = 2 * x_np + 1 + 0.1 * np.random.randn(100, 1).astype(np.float32)

x_t = Tensor(x_np, requires_grad=False)
y_t = Tensor(y_np, requires_grad=False)

model_type = st.selectbox("Optimizer:", ["SGD", "Adam"])
lr = st.slider("Learning rate:", 0.001, 0.5, 0.1)
epochs = st.slider("Epochs:", 10, 200, 100)

if st.button("Train Model"):
    model = Linear(1, 1)
    params = model.parameters()
    opt = SGD(params, lr=lr) if model_type == "SGD" else Adam(params, lr=lr)

    loss_history = []
    for epoch in range(epochs):
        pred = model(x_t)
        loss = mse_loss(pred, y_t)
        opt.zero_grad()
        loss.backward()
        opt.step()
        loss_history.append(loss.data)

    st.success(f"Training completed in {epochs} epochs.")
    st.line_chart(loss_history)
    st.write("Learned weight:", model.weight.data.flatten()[0])
    st.write("Learned bias:", model.bias.data.flatten()[0])

st.markdown("---")
# MNIST classification demo
st.header("MNIST Example")
if st.button("Run MNIST Example"):
    loss_history, accuracy = run_mnist_example()
    st.line_chart(loss_history)
    st.write(f"MNIST Test Accuracy: {accuracy * 100:.2f}%")