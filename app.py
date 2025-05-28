import streamlit as st
import numpy as np
from minitorch.tensor import Tensor
from minitorch.layers import Linear, ReLU, BatchNorm1d
from minitorch.loss import mse_loss
from minitorch.optim import SGD, Adam

st.set_page_config(page_title="MiniTorch Playground", layout="centered")

st.title("MiniTorch Interactive Playground")
st.write("Explore MiniTorch operations, autograd, and training via a simple UI.")

# Section: Tensor Operations
st.header("Tensor Operations")
x_val = st.number_input("Enter a scalar value for x:", value=2.0)
x = Tensor(x_val, requires_grad=True)
y = x * Tensor(3.0) + Tensor(1.0)
y.backward()
st.write(f"Computed y = 3*x + 1 = {y.data}")
st.write(f"Gradient dy/dx = {x.grad}")

st.markdown("---")

# Section: Neural Network Regression Demo
st.header("Train a Tiny Linear Model (y=2x+1)")
# Generate synthetic data
np.random.seed(0)
x_np = np.random.rand(100, 1).astype(np.float32)
y_np = 2 * x_np + 1 + 0.1 * np.random.randn(100, 1).astype(np.float32)

# Convert to MiniTorch Tensors
x_t = Tensor(x_np, requires_grad=False)
y_t = Tensor(y_np, requires_grad=False)

# Model selection
model_type = st.selectbox("Optimizer:", ["SGD", "Adam"])
lr = st.slider("Learning rate:", 0.001, 0.5, 0.1)
epochs = st.slider("Epochs:", 10, 200, 100)

model = Linear(1, 1)
params = model.parameters()
opt = SGD(params, lr=lr) if model_type == "SGD" else Adam(params, lr=lr)

# Train button
if st.button("Train Model"):
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
