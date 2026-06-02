import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

VIDEO_ROOT = Path("video") / "anomalies"
VIDEO_ROOT.mkdir(parents=True, exist_ok=True)

# 身体连接关系（用于单帧画骨架）
CONNECTIONS = [
    ('kpt1', 'kpt2'),   # Nose -> Left Eye
    ('kpt1', 'kpt3'),   # Nose -> Right Eye
    ('kpt2', 'kpt4'),   # Left Eye -> Left Ear
    ('kpt3', 'kpt5'),   # Right Eye -> Right Ear
    ('kpt6', 'kpt7'),   # Left Shoulder <-> Right Shoulder
    ('kpt6', 'kpt8'),   # Left Shoulder -> Left Elbow
    ('kpt8', 'kpt10'),  # Left Elbow -> Left Wrist
    ('kpt7', 'kpt9'),   # Right Shoulder -> Right Elbow
    ('kpt9', 'kpt11'),  # Right Elbow -> Right Wrist
    ('kpt6', 'kpt12'),  # Left Shoulder -> Left Hip
    ('kpt7', 'kpt13'),  # Right Shoulder -> Right Hip
    ('kpt12', 'kpt13'), # Left Hip <-> Right Hip
    ('kpt12', 'kpt14'), # Left Hip -> Left Knee
    ('kpt14', 'kpt16'), # Left Knee -> Left Ankle
    ('kpt13', 'kpt15'), # Right Hip -> Right Knee
    ('kpt15', 'kpt17')  # Right Knee -> Right Ankle
]


def visualize_frame(sample_arr, start_row: int, anomaly_row: int, cols=None, title: str = "Anomaly Frame"):
    """可视化窗口中对应异常的单帧（不保存为视频）。
    sample_arr: ndarray (frames, coords)
    start_row: 原始CSV中该sample_arr对应的起始行索引
    anomaly_row: 原始CSV中出现异常的行索引
    """
    frames, coords = sample_arr.shape
    num_keypoints = min(17, coords // 2)

    rel_idx = anomaly_row - int(start_row)
    if rel_idx < 0:
        rel_idx = 0
    if rel_idx >= frames:
        rel_idx = frames - 1

    frame = sample_arr[rel_idx]

    # 建立 keypoint 名称 -> pair 索引 的映射（如果提供列名则优先使用）
    name_to_pair = {}
    if cols is not None:
        # 尝试按命名规则找到 kptN_x 列
        cols_list = list(cols)
        for n in range(1, 18):
            name = f'kpt{n}'
            x_name = f'{name}_x'
            # 支持大小写变化
            candidates = [x_name, x_name.upper(), x_name.replace('kpt', 'KPT')]
            found = False
            for c in candidates:
                if c in cols_list:
                    idx = cols_list.index(c)
                    pair_idx = idx // 2
                    name_to_pair[name] = pair_idx
                    found = True
                    break
            if not found:
                # 没找到则跳过，后面会尝试位置映射
                continue

    # fallback: 按位置映射 kpt1->pair0, kpt2->pair1, ...
    for n in range(1, num_keypoints + 1):
        name = f'kpt{n}'
        if name not in name_to_pair:
            name_to_pair[name] = n - 1

    # 收集所有关键点坐标用于画框
    kp_coords = {}
    xs_all = []
    ys_all = []
    for name, pair_idx in name_to_pair.items():
        xi = frame[pair_idx * 2] if pair_idx * 2 < len(frame) else np.nan
        yi = frame[pair_idx * 2 + 1] if pair_idx * 2 + 1 < len(frame) else np.nan
        if np.isnan(xi) or np.isnan(yi):
            continue
        kp_coords[name] = (xi, yi)
        xs_all.append(xi)
        ys_all.append(yi)

    if not kp_coords:
        print(f"No valid coords to display for {title} (row {anomaly_row})")
        return

    x_min, x_max = float(np.nanmin(xs_all)), float(np.nanmax(xs_all))
    y_min, y_max = float(np.nanmin(ys_all)), float(np.nanmax(ys_all))

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.set_xlim(x_min - 0.1, x_max + 0.1)
    ax.set_ylim(y_min - 0.1, y_max + 0.1)
    ax.set_aspect('equal')
    ax.invert_yaxis()
    ax.set_title(f"{title} (row {anomaly_row})")

    # 画点
    xs_plot = [v[0] for v in kp_coords.values()]
    ys_plot = [v[1] for v in kp_coords.values()]
    ax.scatter(xs_plot, ys_plot, c='red', s=60)

    # 画骨架连线
    for a, b in CONNECTIONS:
        if a in kp_coords and b in kp_coords:
            xa, ya = kp_coords[a]
            xb, yb = kp_coords[b]
            ax.plot([xa, xb], [ya, yb], 'b-', linewidth=2)

    # 保存图片并展示
    safe_name = title.replace(' ', '_')
    save_path = VIDEO_ROOT / f"{safe_name}_row_{anomaly_row}.png"
    try:
        fig.savefig(save_path)
        print(f"Saved image: {save_path}")
    except Exception as e:
        print(f"Failed to save image {save_path}: {e}")

    plt.show()
    plt.close(fig)


def scan_keypoints_csvs(kp_dir: Path, window=50):
    """扫描目录下所有 CSV 文件，检测值 <0 或 >1，记录位置并保存窗口数据到 samples。"""
    anomalies = []
    samples = {}  # key: (filename, row_idx) -> (start_row, ndarray (frames, coords))

    if not kp_dir.exists():
        print(f"Keypoints directory not found: {kp_dir}")
        return anomalies, samples

    csvs = sorted(kp_dir.glob('*.csv'))
    if not csvs:
        print(f"No CSV files found in {kp_dir}")
        return anomalies, samples

    for p in csvs:
        try:
            df = pd.read_csv(p)
        except Exception as e:
            print(f"Failed to read {p}: {e}")
            continue

        # drop Time column if present
        if 'Time' in df.columns:
            df_coords = df.drop(columns=['Time'], errors='ignore')
        else:
            df_coords = df

        # convert to numeric, coerce errors
        df_coords = df_coords.apply(pd.to_numeric, errors='coerce')

        # find anomalies (<0 or >1)
        mask_low = df_coords < -1
        mask_high = df_coords > 2
        bad_mask = mask_low | mask_high
        if not bad_mask.values.any():
            continue

        rows, cols = np.where(bad_mask.values)
        rows_list = rows.tolist()
        cols_list = cols.tolist()
        # 记录每个异常位置（按单元格）
        for r, c in zip(rows_list, cols_list):
            val = float(df_coords.iat[r, c])
            col_name = df_coords.columns[c]
            anomalies.append({
                'file': str(p),
                'row': int(r),
                'col_name': str(col_name),
                'col_idx': int(c),
                'value': val
            })

        # 仅为每个包含异常的行保存一次样本窗口（避免重复）
        unique_rows = sorted(set(rows_list))
        cols_names = list(df_coords.columns)
        for r in unique_rows:
            start = max(0, r - window // 2)
            end = start + window
            if end > len(df_coords):
                end = len(df_coords)
                start = max(0, end - window)
            sample = df_coords.iloc[start:end].values.astype(float)
            key = (str(p), int(r))
            if key not in samples:
                # store start index, array, and column names
                samples[key] = (start, sample, cols_names)

    return anomalies, samples


def main(argv):
    # determine directory
    if len(argv) > 1:
        kp_dir = Path(argv[1])
    else:
        kp_dir = Path("/rd/zljx/dataset/SWIT/SWIT_Dataset/Keypoints_Raw")

    print(f"Scanning keypoints in: {kp_dir}")
    anomalies, samples = scan_keypoints_csvs(kp_dir, window=50)

    print(f"\nScan complete. Total anomaly entries found: {len(anomalies)}")
    print(f"Total unique samples with anomalies: {len(samples)}")

    if len(anomalies) == 0:
        return

    # 打印前20条示例（按 file,row 去重并汇总同一行的异常列）
    grouped = {}
    for a in anomalies:
        key = (a['file'], a['row'])
        grouped.setdefault(key, []).append(a)

    printed = 0
    for (fpath, row), entries in grouped.items():
        if printed >= 20:
            break
        cols_summary = ", ".join([f"{e['col_name']}={e['value']}" for e in entries])
        print(f"[{printed+1}] file={fpath}, row={row}, cols=[{cols_summary}]")
        printed += 1

    # 询问用户是否可视化
    while True:
        choice = input("可视化选项：输入 'all' 可视化所有样本，输入数字 n 可视化前 n 条样本，输入 'none' 退出：").strip().lower()
        if choice in ('none', 'n'):
            print("退出，不做可视化。")
            return
        if choice == 'all':
            keys = list(samples.keys())
            break
        if choice.isdigit():
            n = int(choice)
            keys = list(samples.keys())[:n]
            break
        print("无效输入，请重试。")

    for key in keys:
        path_str, row_idx = key
        start_row, sample_arr, cols = samples[key]
        fname = Path(path_str).stem
        title = f"{fname} row {row_idx}"
        # 可视化问题帧（显示并保存为 PNG）
        visualize_frame(sample_arr, start_row, row_idx, cols=cols, title=title)


if __name__ == '__main__':
    import os
    main(sys.argv)
