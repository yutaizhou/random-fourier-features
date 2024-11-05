"""============================================================================
Kernel approximation using random Fourier features. Based on "Random Features
for Large-Scale Kernel Machines" by Rahimi and Recht (2007).

For more, see the accompanying blog post:
http://gregorygundersen.com/blog/2019/12/23/random-fourier-features/
============================================================================"""

import einops
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_s_curve
from sklearn.metrics.pairwise import rbf_kernel


def plot_kernel(ax, Z, title, error: float):
    font = {"fontname": "arial", "fontsize": 12}
    ax.imshow(Z, cmap=plt.cm.Blues)
    ax.set_title(f"{title} ({error=:.2f})", **font)
    ax.set_xticks([])
    ax.set_yticks([])


def compute_error(K, K_NN):
    return np.linalg.norm(K - K_NN)


# Generate S-curve data in 3D space
N = 1000
D = 3
X_ND, t = make_s_curve(N, noise=0.1)
X_ND = X_ND[t.argsort()]
print(f"{X_ND.shape = }, {t.shape = }")

# The RBF kernel is the Gaussian kernel if we let \gamma = 1 / (2 \sigma^2).
# We we want to approximate the Gaussian kernel with random Fourier features
K_NN = rbf_kernel(X_ND, gamma=1 / 2.0)

num_fourier_features = [1, 10, 100, 1000, 10000]
fig, axes = plt.subplots(1, 1 + len(num_fourier_features))
fig.set_size_inches(15, 5)


plot_kernel(axes[0], K_NN, "Exact RBF kernel", error=compute_error(K_NN, K_NN))

for R, ax in zip(num_fourier_features, axes[1:]):
    W_RD = np.random.normal(loc=0, scale=1, size=(R, D))
    b_R = np.random.uniform(0, 2 * np.pi, size=R)
    B_RN = einops.repeat(b_R, "b -> b n", n=N)
    Z_RN = np.sqrt(2) * np.cos(W_RD @ X_ND.T + B_RN) / np.sqrt(R)
    ZZ_NN = Z_RN.T @ Z_RN

    approx_error = compute_error(ZZ_NN, K_NN)
    plot_kernel(ax, ZZ_NN, r"$R=%s$" % R, error=approx_error)


plt.tight_layout()
plt.show()
