import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, ListedColormap

# one fixed color per activation / loss / class in every figure
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
LINE_SERIES = SERIES + ["#eda100", "#e87ba4"]  # up to five line series, same fixed order
ACT_COLORS = {"sigmoid": SERIES[0], "tanh": SERIES[1], "relu": SERIES[2]}
CLASS_CMAP = ListedColormap(SERIES)
INK, MUTED, TRAIN_GRAY, TEST_GRAY = "#0b0b0b", "#898781", "#6f6e69", "#d6d4cf"
WEIGHT_CMAP = LinearSegmentedColormap.from_list("weights", ["#e34948", "#f0efec", "#2a78d6"])  # negative, 0, positive
ERROR_CMAP = LinearSegmentedColormap.from_list("errors", ["#ffffff", "#9ec5f4", "#2a78d6", "#104281"])  # 0 -> most

plt.rcParams.update({
    "figure.dpi": 100, "savefig.dpi": 200, "savefig.bbox": "tight",
    "axes.grid": True, "grid.color": "#e6e5e1", "grid.linewidth": 0.6,
    "axes.edgecolor": MUTED, "axes.spines.top": False, "axes.spines.right": False,
    "lines.linewidth": 1.5, "legend.frameon": False,
})


def plot_data_reg(ax, X_train, Y_train, X_test, Y_test):
    ax.scatter(X_test, Y_test, s=3, c=TEST_GRAY, label="test")
    ax.scatter(X_train, Y_train, s=4, c=TRAIN_GRAY, label="train")
    ax.set_xlabel("x")
    ax.set_ylabel("y")


def plot_reg_fit(ax, predict, X_train, Y_train, X_test, Y_test, title, color):
    # predict maps x of shape (1, M) to y of shape (1, M), both in original units
    plot_data_reg(ax, X_train, Y_train, X_test, Y_test)
    x_grid = np.linspace(X_test.min(), X_test.max(), 400).reshape(1, -1)
    ax.plot(x_grid[0], predict(x_grid)[0], c=color, lw=2, label="network")
    for edge in (X_train.min(), X_train.max()):
        ax.axvline(edge, c=MUTED, ls="--", lw=0.8)
    ax.set_title(title)


def plot_data_cls(ax, X, Y):
    ax.scatter(X[0], X[1], c=Y.argmax(axis=0), s=4, cmap=CLASS_CMAP, vmin=0, vmax=len(SERIES) - 1)
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")


def plot_decision(ax, net, X, Y, title):
    # color every point of a dense grid by the class the network predicts there
    xx, yy = np.meshgrid(np.linspace(X[0].min() - 0.2, X[0].max() + 0.2, 250),
                         np.linspace(X[1].min() - 0.2, X[1].max() + 0.2, 250))
    grid = np.vstack([xx.ravel(), yy.ravel()])
    pred = net.predict(grid).argmax(axis=0).reshape(xx.shape)
    ax.contourf(xx, yy, pred, levels=np.arange(len(SERIES) + 1) - 0.5, cmap=CLASS_CMAP, alpha=0.25)
    plot_data_cls(ax, X, Y)
    ax.set_title(title)


def plot_loss(ax, history, label=None, color=INK):
    ax.plot(np.arange(1, len(history["loss"]) + 1), history["loss"], c=color, label=label)
    ax.set_yscale("log")
    ax.set_xlabel("epoch")
    ax.set_ylabel("training loss")


def plot_weights(history, title):
    # every line is one weight; one panel per layer
    n_layers = len(history["weights"][0])
    fig, axes = plt.subplots(1, n_layers, figsize=(5 * n_layers, 3.4), squeeze=False)
    for l, ax in enumerate(axes[0]):
        values = np.array([w[l].ravel() for w in history["weights"]])  # (epochs, weights in layer)
        ax.plot(np.arange(1, len(values) + 1), values, c=SERIES[0], lw=0.8, alpha=0.7)
        ax.set_title(f"layer {l + 1}: W{l + 1} {history['weights'][0][l].shape}")
        ax.set_xlabel("epoch")
        ax.set_ylabel("weight value")
    fig.suptitle(title)
    fig.tight_layout()
    return fig


def plot_digits(X, titles, ncols=10):
    # X: (784, n), one 28x28 image per column
    nrows = -(-X.shape[1] // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(1.1 * ncols, 1.3 * nrows), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for ax, image, title in zip(axes.ravel(), X.T, titles):
        ax.imshow(image.reshape(28, 28), cmap="gray_r", vmin=0, vmax=1)
        ax.set_title(title, fontsize=9)
    fig.tight_layout()
    return fig


def plot_confusion(ax, confusion, title):
    # rows: true class, columns: predicted class; color shows only the errors, the diagonal is written in bold
    errors = confusion.astype(float)
    np.fill_diagonal(errors, np.nan)
    ax.imshow(errors, cmap=ERROR_CMAP, vmin=0)
    ax.grid(False)
    K = len(confusion)
    for i in range(K):
        for j in range(K):
            dark = i != j and errors[i, j] > 0.6 * np.nanmax(errors)
            ax.text(j, i, int(confusion[i, j]), ha="center", va="center", fontsize=8,
                    color="white" if dark else INK, fontweight="bold" if i == j else "normal")
    ax.set_xticks(range(K))
    ax.set_yticks(range(K))
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(title)


def plot_network(history, epochs, title, ncols=2):
    # the network as layers of nodes; every edge is one weight, shown at the given epochs
    snapshots = [history["weights"][e - 1] for e in epochs]
    sizes = [snapshots[0][0].shape[1]] + [W.shape[0] for W in snapshots[0]]
    scale = max(np.abs(W).max() for weights in snapshots for W in weights)  # same scale for every epoch
    pos = [[(l, (n - 1) / 2 - i) for i in range(n)] for l, n in enumerate(sizes)]

    nrows = -(-len(epochs) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, (0.5 * max(sizes) + 1) * nrows), squeeze=False)
    for ax in axes.ravel()[len(epochs):]:
        ax.axis("off")
    for ax, epoch, weights in zip(axes.ravel(), epochs, snapshots):
        for l, W in enumerate(weights):
            n_out, n_in = W.shape
            for i in range(n_out):
                for j in range(n_in):
                    (x0, y0), (x1, y1) = pos[l][j], pos[l + 1][i]
                    w = W[i, j]
                    # labels go on the side where the edges spread out, one column per node on the narrow side
                    if n_in < n_out:
                        t = 0.75 - 0.4 * j / max(n_in - 1, 1)
                    else:
                        t = 0.25 + 0.4 * i / max(n_out - 1, 1)
                    ax.plot([x0, x1], [y0, y1], c=WEIGHT_CMAP(0.5 + w / (2 * scale)),
                            lw=0.5 + 4 * abs(w) / scale, zorder=1)
                    ax.text(x0 + t * (x1 - x0), y0 + t * (y1 - y0), f"{w:.2f}", fontsize=7.5, color=INK,
                            ha="center", va="center", zorder=3,
                            bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.85))
        for layer in pos:
            ax.scatter(*zip(*layer), s=160, c="white", edgecolors=INK, linewidths=1, zorder=2)
        ax.set_title(f"epoch {epoch}")
        ax.axis("off")
    fig.suptitle(f"{title}   (blue: positive, red: negative, line width: |w|)")
    fig.tight_layout()
    return fig
