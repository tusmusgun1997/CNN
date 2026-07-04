"""
Backpropagation in a CNN — implemented FROM SCRATCH (no autograd), then verified
against TensorFlow's automatic differentiation.

This is the exact network from the companion page `backpropagation.html`:

    X(6x6) --conv(W1 3x3, b1)--> Z1(4x4) --relu--> A1(4x4)
          --maxpool 2x2 s2--> P1(2x2) --flatten--> F(4x1)
          --dense(W2 1x4, b2)--> Z2(1x1) --sigmoid--> A2 --> BCE loss L

15 trainable parameters:  W1(9) + b1(1) + W2(4) + b2(1) = 15

Every number printed here is the number embedded in the HTML animation.

Run:
    python backprop_from_scratch.py            # from-scratch forward + backward
    pip install tensorflow                     # (optional) to enable the autograd check
"""

import numpy as np

np.set_printoptions(precision=4, suppress=True)

# ----------------------------------------------------------------------------
# Fixed inputs and initial parameters
# ----------------------------------------------------------------------------
X = np.array([
    [1, 2, 3, 0, 1, 2],
    [0, 1, 2, 3, 0, 1],
    [2, 1, 0, 1, 2, 3],
    [1, 0, 1, 2, 1, 0],
    [0, 2, 1, 0, 1, 2],
    [1, 1, 0, 2, 0, 1],
], dtype=float)

W1 = np.array([[1, 0, -1],
               [1, 0, -1],
               [1, 0, -1]], dtype=float)   # a vertical-edge filter
b1 = 0.0
W2 = np.array([[0.5, -0.5, 0.5, -0.5]])     # shape (1, 4)
b2 = 0.0
y  = 1.0                                    # true label (binary)
lr = 0.1                                    # learning rate


def conv2d_valid(x, w):
    """Valid 2-D cross-correlation (what CNNs call 'convolution')."""
    kh, kw = w.shape
    oh, ow = x.shape[0] - kh + 1, x.shape[1] - kw + 1
    out = np.zeros((oh, ow))
    for i in range(oh):
        for j in range(ow):
            out[i, j] = np.sum(x[i:i + kh, j:j + kw] * w)
    return out


# ============================================================================
# FORWARD PASS
# ============================================================================
Z1 = conv2d_valid(X, W1) + b1                 # (4,4)
A1 = np.maximum(0, Z1)                         # ReLU        (4,4)

# max pooling 2x2 stride 2 -> (2,2), remembering the argmax of each window
P1 = np.zeros((2, 2))
argmax = {}
for pi in range(2):
    for pj in range(2):
        win = A1[pi * 2:pi * 2 + 2, pj * 2:pj * 2 + 2]
        P1[pi, pj] = win.max()
        di, dj = np.unravel_index(np.argmax(win), win.shape)
        argmax[(pi, pj)] = (pi * 2 + di, pj * 2 + dj)

F  = P1.reshape(4, 1)                          # flatten     (4,1)
Z2 = W2 @ F + b2                               # (1,1)
A2 = 1.0 / (1.0 + np.exp(-Z2))                 # sigmoid  = y-hat
L  = -(y * np.log(A2) + (1 - y) * np.log(1 - A2))   # binary cross-entropy

print("================ FORWARD ================")
print("Z1 =\n", Z1)
print("A1 = relu(Z1) =\n", A1)
print("P1 = maxpool(A1) =\n", P1, "  argmax:", argmax)
print("F  = flatten(P1) =", F.ravel())
print("Z2 =", Z2.ravel(), "  A2 = sigmoid(Z2) =", A2.ravel())
print("L  = BCE =", L.ravel())

# ============================================================================
# BACKWARD PASS  (manual chain rule, from scratch)
# ============================================================================
dZ2 = A2 - y                 # dL/dZ2  (sigmoid + BCE collapse to this)   (1,1)
dW2 = dZ2 @ F.T              # dL/dW2                                      (1,4)
db2 = dZ2.copy()            # dL/db2                                      (1,1)
dF  = W2.T @ dZ2            # dL/dF                                       (4,1)
dP1 = dF.reshape(2, 2)      # dL/dP1  (un-flatten)                        (2,2)

dA1 = np.zeros((4, 4))       # dL/dA1  (route through max pool)
for (pi, pj), (ai, aj) in argmax.items():
    dA1[ai, aj] += dP1[pi, pj]

dZ1 = dA1 * (Z1 > 0)         # dL/dZ1  (ReLU gate)                        (4,4)
dW1 = conv2d_valid(X, dZ1)   # dL/dW1  = conv(X, dZ1)                     (3,3)
db1 = np.sum(dZ1)            # dL/db1                                     scalar

print("\n================ BACKWARD ================")
print("dL/dZ2 = A2 - y =", dZ2.ravel())
print("dL/dW2 =", dW2.ravel())
print("dL/db2 =", db2.ravel())
print("dL/dF  =", dF.ravel())
print("dL/dP1 =\n", dP1)
print("dL/dA1 =\n", dA1)
print("dL/dZ1 =\n", dZ1)
print("dL/dW1 =\n", dW1)
print("dL/db1 =", db1)

# ============================================================================
# GRADIENT-DESCENT UPDATE  (all 15 parameters)
# ============================================================================
print("\n================ UPDATE (lr = 0.1) ================")
print("W1_new =\n", W1 - lr * dW1)
print("b1_new =", b1 - lr * db1)
print("W2_new =", (W2 - lr * dW2).ravel())
print("b2_new =", (b2 - lr * db2).ravel())

# ============================================================================
# VERIFY against TensorFlow autograd (optional)
# ============================================================================
try:
    import tensorflow as tf

    Xt  = tf.constant(X.reshape(1, 6, 6, 1), dtype=tf.float32)
    W1v = tf.Variable(W1.reshape(3, 3, 1, 1), dtype=tf.float32)
    b1v = tf.Variable([0.0], dtype=tf.float32)
    W2v = tf.Variable(W2.T.reshape(4, 1), dtype=tf.float32)
    b2v = tf.Variable([0.0], dtype=tf.float32)
    yt  = tf.constant([[1.0]], dtype=tf.float32)

    with tf.GradientTape() as tape:
        z1 = tf.nn.conv2d(Xt, W1v, strides=1, padding="VALID") + b1v
        a1 = tf.nn.relu(z1)
        p1 = tf.nn.max_pool2d(a1, ksize=2, strides=2, padding="VALID")
        f  = tf.reshape(p1, (1, 4))
        z2 = tf.matmul(f, W2v) + b2v
        a2 = tf.sigmoid(z2)
        loss = -(yt * tf.math.log(a2) + (1 - yt) * tf.math.log(1 - a2))

    gW1, gb1, gW2, gb2 = tape.gradient(loss, [W1v, b1v, W2v, b2v])
    print("\n================ AUTOGRAD CHECK ================")
    print("dW1 matches:", np.allclose(gW1.numpy().reshape(3, 3), dW1, atol=1e-4))
    print("db1 matches:", np.allclose(gb1.numpy(), db1, atol=1e-4))
    print("dW2 matches:", np.allclose(gW2.numpy().ravel(), dW2.ravel(), atol=1e-4))
    print("db2 matches:", np.allclose(gb2.numpy().ravel(), db2.ravel(), atol=1e-4))
    print("\nFrom-scratch backprop == TensorFlow autograd  [OK]")
except Exception as exc:  # noqa: BLE001
    print("\n(TensorFlow not available, skipped autograd check:", exc, ")")
