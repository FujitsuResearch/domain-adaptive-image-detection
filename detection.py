#!/usr/bin/env python3
"""
Zero-shot generated image detection using conditional likelihood approximation
with CLIP embeddings and whitening.
"""

import os
import argparse
from typing import List, Tuple, Sequence

import numpy as np
import torch
from tqdm import tqdm
from PIL import Image
import clip  # pip install git+https://github.com/openai/CLIP.git
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    accuracy_score,
)

IMAGE_EXTS = (".png", ".jpg", ".jpeg")


# ---------------------------------------------------------------------------
#  Core linear algebra: whitening / sphering
# ---------------------------------------------------------------------------


def sphx(x: torch.Tensor, m: int = 500) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Whitening transformation (PCA-based sphering).

    Args:
        x: [N, D] tensor of embeddings.
        m: number of principal components to keep (<= D).

    Returns:
        sph: [N, m] whitened embeddings.
        rotation_matrix: [D, m] whitening matrix.
    """
    assert x.ndim == 2, f"Expected 2D tensor, got {x.ndim}D"
    _, d = x.shape
    m = min(m, d)

    device = x.device
    x_mean = x.mean(dim=0)
    xu = x - x_mean  # [N, D]

    # Covariance on the same device as x
    cov_matrix = torch.cov(xu.T).to(device)  # [D, D] (symmetric PSD)

    eigenvalues, eigenvectors = torch.linalg.eigh(cov_matrix)
    # Sort eigenvalues descending and take top-m
    indices = torch.argsort(eigenvalues, descending=True)[:m]

    s_matrix = torch.diag(eigenvalues[indices])  # [m, m]
    v_matrix = eigenvectors[:, indices]  # [D, m]

    # S^{-1/2}
    s_inv_sqrt = torch.sqrt(torch.inverse(s_matrix))
    rotation_matrix = v_matrix @ s_inv_sqrt  # [D, m]

    sph = xu @ rotation_matrix  # [N, m]
    return sph, rotation_matrix


# ---------------------------------------------------------------------------
#  Utilities
# ---------------------------------------------------------------------------


def list_image_files(directory: str) -> List[str]:
    """Return sorted list of image file paths in a directory."""
    files = [
        os.path.join(directory, f)
        for f in os.listdir(directory)
        if f.lower().endswith(IMAGE_EXTS)
    ]
    files.sort()
    return files


def load_clip_model(device: str = "cuda") -> Tuple[torch.nn.Module, callable]:
    """Return CLIP model with preprocess function."""
    model, preprocess = clip.load("ViT-L/14", device=device)
    model.eval()
    return model, preprocess


# ---------------------------------------------------------------------------
#  Embedding and whitening set construction
# ---------------------------------------------------------------------------


def create_rep_set(
    image_dir: str,
    model: torch.nn.Module,
    preprocess,
    device: str,
    output_path: str,
) -> torch.Tensor:
    """
    Embed all images in `image_dir` using CLIP and save embeddings as a tensor.

    Returns the embeddings tensor [N, D].
    """
    embeddings: List[torch.Tensor] = []
    image_files = list_image_files(image_dir)

    for image_file in tqdm(image_files, desc="Embedding rep set"):
        img = Image.open(image_file).convert("RGB")
        image_tensor = preprocess(img).unsqueeze(0).to(device)
        with torch.no_grad():
            embedding = (
                model.encode_image(image_tensor)
                .squeeze(0)
                .to(device, dtype=torch.float32)
            )
        embeddings.append(embedding.cpu())

    if not embeddings:
        raise ValueError(f"No images found in {image_dir}")

    embeddings_tensor = torch.stack(embeddings)  # [N, D]
    torch.save(embeddings_tensor, output_path)
    print(f"Saved rep set embeddings {embeddings_tensor.shape} to {output_path}")
    return embeddings_tensor


def create_w_mat(
    rep_dir: str,
    model: torch.nn.Module,
    preprocess,
    device: str,
    output_path: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute a global whitening matrix from images in `rep_dir` and save it.

    Saves (w_mat, w_mean) where:
        w_mat: [D, m] whitening matrix (here m == D)
        w_mean: [D] mean embedding
    """
    rep_embeddings = create_rep_set(
        rep_dir, model, preprocess, device, output_path + ".tmp"
    )
    os.remove(output_path + ".tmp")

    rep_embeddings = rep_embeddings.to(device)
    _, w_mat = sphx(rep_embeddings, m=rep_embeddings.shape[1])
    w_mean = rep_embeddings.mean(dim=0)

    torch.save((w_mat.cpu(), w_mean.cpu()), output_path)
    print(f"Saved whitening matrix and mean to {output_path}")
    return w_mat, w_mean


# ---------------------------------------------------------------------------
#  Likelihood computation
# ---------------------------------------------------------------------------


def likelihood_from_rep(
    image_dir: str,
    rep_mat: torch.Tensor,
    k: int,
    m: int,
    model: torch.nn.Module,
    preprocess,
    device: str,
    output_path: str,
):
    """
    Per-image local whitening using k nearest rep samples.

    Args:
        image_dir: directory of images to score.
        rep_mat: [N_rep, D] representative embeddings (on `device`).
        k: number of neighbors for local whitening.
        m: number of PCA components for whitening.
    """
    likelihoods: List[torch.Tensor] = []
    image_files = list_image_files(image_dir)

    rep_mat = rep_mat.to(device)
    log_const = 0.5 * m * torch.log(torch.tensor(2 * np.pi, device=device))

    for image_file in tqdm(image_files, desc="Computing likelihoods (local whitening)"):
        img = Image.open(image_file).convert("RGB")
        image_tensor = preprocess(img).unsqueeze(0).to(device)
        with torch.no_grad():
            embedding = (
                model.encode_image(image_tensor)
                .squeeze(0)
                .to(device, dtype=torch.float32)
            )

        # cosine similarity to rep set
        similarities = torch.cosine_similarity(embedding, rep_mat, dim=1)  # [N_rep]
        top_k_indices = torch.topk(similarities, k=k, largest=True).indices
        selected_rep = rep_mat[top_k_indices]  # [k, D]

        _, local_w = sphx(selected_rep, m=m)  # [D, m]
        whitened_embedding = (embedding - selected_rep.mean(dim=0)) @ local_w  # [m]

        likelihood = -(log_const + 0.5 * whitened_embedding.norm() ** 2)
        likelihoods.append(likelihood.detach().cpu())

    likelihoods_tensor = torch.stack(likelihoods, dim=0)  # [N_images]
    torch.save(likelihoods_tensor, output_path)
    print(f"Saved likelihoods to {output_path}")


def likelihood_from_mat(
    image_dir: str,
    w_mat: torch.Tensor,
    w_mean: torch.Tensor,
    model: torch.nn.Module,
    preprocess,
    device: str,
    output_path: str,
):
    """
    Likelihood using a precomputed global whitening matrix.

    Args:
        image_dir: directory of images to score.
        w_mat: [D, m] whitening matrix.
        w_mean: [D] mean embedding used to compute w_mat.
    """
    likelihoods: List[torch.Tensor] = []
    image_files = list_image_files(image_dir)

    w_mat = w_mat.to(device)
    w_mean = w_mean.to(device)
    m = w_mat.shape[1]
    log_const = 0.5 * m * torch.log(torch.tensor(2 * np.pi, device=device))

    for image_file in tqdm(
        image_files, desc="Computing likelihoods (global whitening)"
    ):
        img = Image.open(image_file).convert("RGB")
        image_tensor = preprocess(img).unsqueeze(0).to(device)
        with torch.no_grad():
            embedding = (
                model.encode_image(image_tensor)
                .squeeze(0)
                .to(device, dtype=torch.float32)
            )

        whitened_embedding = (embedding - w_mean) @ w_mat
        likelihood = -(log_const + 0.5 * whitened_embedding.norm() ** 2)
        likelihoods.append(likelihood.detach().cpu())

    likelihoods_tensor = torch.stack(likelihoods, dim=0)
    torch.save(likelihoods_tensor, output_path)
    print(f"Saved likelihoods to {output_path}")


# ---------------------------------------------------------------------------
#  Metrics
# ---------------------------------------------------------------------------


def likelihood_to_metrics(
    likelihoods: torch.Tensor,
    labels: Sequence[int],
    threshold: float,
    output_path: str,
):
    """
    Compute AUC, AP, F1-score, and Accuracy from likelihoods and labels.

    Args:
        likelihoods: [N] tensor of likelihood scores.
        labels: iterable of 0/1 labels.
        threshold: scalar threshold on likelihood for binary prediction.
    """
    likelihoods_np = likelihoods.numpy()
    labels_np = np.asarray(labels)

    auc = roc_auc_score(labels_np, likelihoods_np)
    ap = average_precision_score(labels_np, likelihoods_np)

    preds = (likelihoods_np > threshold).astype(int)
    f1 = f1_score(labels_np, preds)
    acc = accuracy_score(labels_np, preds)

    metrics = {"AUC": auc, "AP": ap, "F1": f1, "Accuracy": acc}
    torch.save(metrics, output_path)
    print(f"Saved metrics to {output_path}")
    return metrics


# ---------------------------------------------------------------------------
#  Main / CLI
# ---------------------------------------------------------------------------


def main(args):
    "Main function of the detection script - calculates likelihoods and metrics."
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    model, preprocess = load_clip_model(device=device)

    # Load labels and threshold
    if not os.path.exists(args.labels_path):
        raise FileNotFoundError(f"Labels file not found at {args.labels_path}")
    labels, threshold = torch.load(args.labels_path, weights_only=False)

    # Compute likelihoods if needed
    if not os.path.exists(args.likelihood_path):
        if args.use_global:
            # global whitening matrix route
            if not os.path.exists(args.w_mat_path):
                print(f"Computing global whitening matrix into {args.w_mat_path}...")
                create_w_mat(
                    args.rep_dir_path, model, preprocess, device, args.w_mat_path
                )

            w_mat, w_mean = torch.load(args.w_mat_path, weights_only=False)
            likelihood_from_mat(
                args.image_path,
                w_mat,
                w_mean,
                model,
                preprocess,
                device,
                args.likelihood_path,
            )
        else:
            # local whitening route
            if not os.path.exists(args.rep_mat_path):
                print(
                    f"Computing representative set matrix into {args.rep_mat_path}..."
                )
                rep_embeddings = create_rep_set(
                    args.rep_dir_path, model, preprocess, device, args.rep_mat_path
                )
            else:
                rep_embeddings = torch.load(args.rep_mat_path, weights_only=False)

            rep_embeddings = rep_embeddings.to(device)
            likelihood_from_rep(
                args.image_path,
                rep_embeddings,
                args.k,
                args.m,
                model,
                preprocess,
                device,
                args.likelihood_path,
            )

    likelihoods = torch.load(args.likelihood_path, weights_only=False)
    metrics = likelihood_to_metrics(likelihoods, labels, threshold, args.metrics_path)
    print(metrics)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Zero-shot generated image detection using "
            "conditional likelihood approximation."
        )
    )

    parser.add_argument(
        "--image_path",
        type=str,
        default="image_dir",
        help="Directory with images for evaluation.",
    )
    parser.add_argument(
        "--rep_dir_path",
        type=str,
        default="rep_dir",
        help="Directory with representative images (real).",
    )
    parser.add_argument(
        "--labels_path",
        type=str,
        default="labels.pt",
        help="Path to the (labels, threshold) tuple saved with torch.save.",
    )
    parser.add_argument(
        "--w_mat_path",
        type=str,
        default="whitening_matrix.pt",
        help="Path to the precomputed whitening matrix (for --use_global).",
    )
    parser.add_argument(
        "--rep_mat_path",
        type=str,
        default="rep_set.pt",
        help="Path to the representative set embeddings.",
    )
    parser.add_argument(
        "--likelihood_path",
        type=str,
        default="likelihoods.pt",
        help="Path to save likelihood values.",
    )
    parser.add_argument(
        "--metrics_path",
        type=str,
        default="metrics.pt",
        help="Path to save evaluation metrics.",
    )

    parser.add_argument(
        "--use_global",
        action="store_true",
        help="Use global whitening matrix instead of per-image local whitening.",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=500,
        help="Number of neighbors used for local whitening.",
    )
    parser.add_argument(
        "--m", type=int, default=400, help="Number of dimensions used for whitening."
    )

    args_ = parser.parse_args()
    main(args_)
