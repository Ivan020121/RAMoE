import argparse
from pathlib import Path
import torch
import numpy as np
import matplotlib.pyplot as plt


def find_keypoint_tensor(loaded):
    """Try to find the keypoint tensor inside the loaded object.
    Returns a torch.Tensor or None.
    """
    # If it's a tensor directly
    if isinstance(loaded, torch.Tensor):
        return loaded

    # If it's a tuple/list/dict, try to locate the first suitable tensor
    if isinstance(loaded, (list, tuple)):
        for item in loaded:
            if isinstance(item, torch.Tensor) and item.ndim >= 2:
                return item
    if isinstance(loaded, dict):
        for v in loaded.values():
            if isinstance(v, torch.Tensor) and v.ndim >= 2:
                return v

    return None


def extract_xy_from_tensor(tensor):
    """Given a tensor of shape (N, T, C) or (T, C) or (N, C),
    extract x and y coordinate lists (flattened) while ignoring NaNs.
    Assumes coordinates are arranged as [x0, y0, x1, y1, ..., xK, yK, ...].
    """
    arr = tensor.detach().cpu().numpy()
    if arr.ndim == 3:
        # (samples, frames, coords)
        flat = arr.reshape(-1, arr.shape[-1])  # (samples*frames, coords)
    elif arr.ndim == 2:
        # (frames, coords) or (samples, coords)
        flat = arr
    else:
        # Unexpected shape
        flat = arr.reshape(-1, arr.shape[-1])

    # Only consider full x,y pairs
    coords = flat.shape[-1]
    pair_count = coords // 2
    use_len = pair_count * 2
    flat = flat[:, :use_len]

    # even indices -> x, odd -> y
    xs = flat[:, 0:use_len:2].ravel()
    ys = flat[:, 1:use_len:2].ravel()

    # Filter NaNs
    xs = xs[~np.isnan(xs)]
    ys = ys[~np.isnan(ys)]

    return xs, ys


def process_dataset(dataset_dir: Path):
    pt_files = sorted(dataset_dir.glob('*.pt'))
    all_x = []
    all_y = []
    per_file_data = []  # list of (Path, xs, ys)

    if not pt_files:
        print(f"No .pt files found in {dataset_dir}")
        return None

    for p in pt_files:
        try:
            loaded = torch.load(p)
        except Exception as e:
            print(f"Failed to load {p}: {e}")
            continue

        tensor = loaded[1]
        if tensor is None:
            print(f"No suitable tensor found in {p}, skipping")
            continue

        xs, ys = extract_xy_from_tensor(tensor)
        if xs.size:
            all_x.append(xs)
        if xs.size:
            # store per-file arrays
            per_file_x = xs
        else:
            per_file_x = np.array([])
        if ys.size:
            all_y.append(ys)
        if ys.size:
            per_file_y = ys
        else:
            per_file_y = np.array([])

        per_file_data.append((p, per_file_x, per_file_y))

    if not all_x or not all_y:
        print("No valid x/y coordinates found in dataset files.")
        return None

    all_x = np.concatenate(all_x)
    all_y = np.concatenate(all_y)

    return all_x, all_y, per_file_data


def plot_and_save(arr, axis_name: str, out_dir: Path, bins=200):
    """Plot a line chart where x-axis is coordinate value and y-axis is
    the proportion of samples falling into the corresponding bin.
    The proportions sum to 1 across all bins.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    # Compute histogram counts (not density) so we can convert to proportions
    counts, bin_edges = np.histogram(arr, bins=bins)
    total = counts.sum()
    if total == 0:
        print(f"No data to plot for {axis_name}")
        return

    proportions = counts.astype(float) / float(total)
    # Use bin centers for plotting
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(bin_centers, proportions, '-o', markersize=3, color='tab:blue')
    ax.set_title(f"Proportion distribution of {axis_name} coordinates (by bin)")
    ax.set_xlabel(f"{axis_name} value")
    ax.set_ylabel("Proportion of values")

    # Stats
    mean = float(np.mean(arr))
    std = float(np.std(arr))
    mn = float(np.min(arr))
    mx = float(np.max(arr))
    ax.text(0.95, 0.95, f"mean={mean:.3f}\nstd={std:.3f}\nmin={mn:.3f}\nmax={mx:.3f}",
            transform=ax.transAxes, ha='right', va='top', bbox=dict(boxstyle='round', fc='white', alpha=0.8))

    ax.grid(True, linestyle='--', alpha=0.4)
    out_path = out_dir / f'kp_{axis_name.lower()}_distribution.png'
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {axis_name} distribution (proportions) to {out_path}")


def main():
    parser = argparse.ArgumentParser(description='Compute and plot KP x/y coordinate distributions')
    parser.add_argument('--dataset', type=str, default='dataset', help='Dataset directory containing .pt files')
    parser.add_argument('--out', type=str, default='viz', help='Output directory for plots')
    parser.add_argument('--bins', type=int, default=200, help='Number of histogram bins')
    parser.add_argument('--save-npz', action='store_true', help='Save aggregated x/y arrays as npz')
    parser.add_argument('--per-file', action='store_true', help='Also generate per-pt-file plots (saved under --out/<pt_stem>/)')
    args = parser.parse_args()

    dataset_dir = Path(args.dataset)
    out_dir = Path(args.out)

    res = process_dataset(dataset_dir)
    if res is None:
        return

    all_x, all_y, per_file_data = res

    # Plot and save separate figures for X and Y (global across all files)
    plot_and_save(all_x, 'X', out_dir, bins=args.bins)
    plot_and_save(all_y, 'Y', out_dir, bins=args.bins)

    # Per-file plotting if requested
    if args.per_file:
        for p, xs, ys in per_file_data:
            subdir = out_dir / p.stem
            # Only plot if there is data
            if xs.size:
                plot_and_save(xs, 'X', subdir, bins=args.bins)
            else:
                print(f"No X data for {p}, skipping per-file X plot")
            if ys.size:
                plot_and_save(ys, 'Y', subdir, bins=args.bins)
            else:
                print(f"No Y data for {p}, skipping per-file Y plot")

            if args.save_npz:
                npz_path = subdir / 'kp_xy_arrays.npz'
                subdir.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(npz_path, x=xs, y=ys)
                print(f"Saved per-file arrays to {npz_path}")

    # Optionally save combined arrays
    if args.save_npz:
        npz_path = out_dir / 'kp_xy_arrays.npz'
        np.savez_compressed(npz_path, x=all_x, y=all_y)
        print(f"Saved combined arrays to {npz_path}")


if __name__ == '__main__':
    main()
