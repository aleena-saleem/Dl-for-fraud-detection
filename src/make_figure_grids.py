"""
Consolidates the full sweep of confusion-matrix PNGs into a small number of
grid figures (one per loss function, all ratios x both models), so your
report/README references 3-4 clean images instead of 24 loose files.

Also moves all the individual per-run PNGs into results/figures/archive/ so
they're kept on disk but out of the way, and out of what you'd normally
commit to GitHub.

Run from the project root:
    python src/make_figure_grids.py
"""

import os
import re
import shutil
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

FIG_DIR = os.path.join("results", "figures")
ARCHIVE_DIR = os.path.join(FIG_DIR, "archive")
GRID_DIR = os.path.join(FIG_DIR, "grids")

MODELS = ["mlp", "mlp_attention"]
LOSSES = ["bce", "weighted_bce", "focal"]
RATIOS = ["1to1", "1to10", "1to100", "full"]


def find_source(fname):
    """Look for a source PNG in figures/ first, then figures/archive/, so the
    script is safe to re-run even after a previous run already archived the
    originals (idempotent)."""
    for d in (FIG_DIR, ARCHIVE_DIR):
        fpath = os.path.join(d, fname)
        if os.path.exists(fpath):
            return fpath
    return None


def build_confusion_grid(loss):
    """One grid per loss: rows = model, cols = ratio."""
    fig, axes = plt.subplots(len(MODELS), len(RATIOS), figsize=(4 * len(RATIOS), 4 * len(MODELS)))

    for i, model in enumerate(MODELS):
        for j, ratio in enumerate(RATIOS):
            ax = axes[i][j]
            fname = f"{model}_{loss}_{ratio}_confusion.png"
            fpath = find_source(fname)
            if fpath:
                img = mpimg.imread(fpath)
                ax.imshow(img)
            else:
                ax.text(0.5, 0.5, "missing", ha="center", va="center")
            ax.set_xticks([]); ax.set_yticks([])
            if i == 0:
                ax.set_title(ratio)
            if j == 0:
                ax.set_ylabel(model, fontsize=12)

    fig.suptitle(f"Confusion matrices — loss={loss}", fontsize=14)
    fig.tight_layout()
    os.makedirs(GRID_DIR, exist_ok=True)
    out_path = os.path.join(GRID_DIR, f"confusion_grid_{loss}.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")


def build_shap_grid():
    """One grid combining TP / FP / FN SHAP summaries for the best run found."""
    labels = ["true_positives", "false_positives", "false_negatives"]

    # find whichever run(s) have shap figures already generated, checking both
    # figures/ and figures/archive/ so this works whether or not a previous
    # run already archived the originals
    shap_runs = set()
    for d in (FIG_DIR, ARCHIVE_DIR):
        if not os.path.isdir(d):
            continue
        for f in os.listdir(d):
            m = re.match(r"(.+)_shap_(true_positives|false_positives|false_negatives)\.png", f)
            if m:
                shap_runs.add(m.group(1))

    for run_name in shap_runs:
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        for ax, label in zip(axes, labels):
            fpath = find_source(f"{run_name}_shap_{label}.png")
            if fpath:
                img = mpimg.imread(fpath)
                ax.imshow(img)
            else:
                ax.text(0.5, 0.5, "missing", ha="center", va="center")
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(label.replace("_", " "))
        fig.suptitle(f"SHAP failure analysis — {run_name}", fontsize=14)
        fig.tight_layout()
        os.makedirs(GRID_DIR, exist_ok=True)
        out_path = os.path.join(GRID_DIR, f"shap_grid_{run_name}.png")
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"Saved {out_path}")


def archive_originals():
    os.makedirs(ARCHIVE_DIR, exist_ok=True)
    moved = 0
    for f in os.listdir(FIG_DIR):
        fpath = os.path.join(FIG_DIR, f)
        if os.path.isfile(fpath) and f.endswith(".png"):
            shutil.move(fpath, os.path.join(ARCHIVE_DIR, f))
            moved += 1
    print(f"Archived {moved} individual figures to {ARCHIVE_DIR}/ "
          f"(0 is fine if this script already ran before)")


def main():
    for loss in LOSSES:
        build_confusion_grid(loss)
    build_shap_grid()
    archive_originals()
    print("\nDone. Keep results/figures/grids/ for your README and repo. "
          "results/figures/archive/ can stay local (add it to .gitignore) "
          "or be deleted once you're happy with the grids.")


if __name__ == "__main__":
    main()
