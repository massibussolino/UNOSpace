"""
UNOPose inference on the Panoptes dataset.

Modes
-----
Single sample (default):
    PYTHONPATH=. python panoptes_infer.py \\
        --config-file configs/main_cfg.py \\
        --ckpt ./weights/model_final.pth \\
        --idx 10

Trajectory (whole dataset as a sequential trajectory):
    PYTHONPATH=. python panoptes_infer.py \\
        --config-file configs/main_cfg.py \\
        --ckpt ./weights/model_final.pth \\
        --trajectory \\
        [--start-idx 1]  [--end-idx N]  [--plot-path traj.png]

Or via the shell wrapper:
    ./core/unopose/panoptes_infer.sh configs/main_cfg.py 0 ./weights/model_final.pth --trajectory
"""

import os

os.environ["PYOPENGL_PLATFORM"] = "egl"

import sys
import os.path as osp
import argparse
import logging

import json

import numpy as np
import torch
from scipy.spatial.transform import Rotation
from tqdm import tqdm

cur_dir = osp.dirname(osp.abspath(__file__))
sys.path.insert(0, cur_dir)

from detectron2.config import LazyConfig, instantiate
from core.unopose.utils.my_checkpoint import MyCheckpointer

from database import PanoptesDataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="UNOPose inference on the Panoptes dataset")
    parser.add_argument("--config-file", required=True, help="Path to the LazyConfig .py file")
    parser.add_argument("--ckpt", required=True, help="Path to the model checkpoint (.pth)")
    parser.add_argument(
        "--dataset-path",
        default=osp.join(cur_dir, "panoptes-datasets/integral"),
        help="Root path of the Panoptes dataset (must contain images/, depths/, masks/)",
    )

    # single-sample mode
    parser.add_argument("--idx", type=int, default=0, help="Dataset index for single-sample inference")

    # trajectory mode
    parser.add_argument("--trajectory", action="store_true", help="Run inference over the full trajectory")
    parser.add_argument("--template-offset", type=int, default=5,
                        help="Frame offset used as template: dataset[idx] uses dataset[idx - K] as reference (default: 5)")
    parser.add_argument("--start-idx", type=int, default=None,
                        help="First frame index for trajectory (default: template_offset)")
    parser.add_argument("--end-idx", type=int, default=None,
                        help="Last frame index (exclusive) for trajectory (default: full dataset)")
    parser.add_argument("--plot-path", type=str, default="trajectory.png",
                        help="Output path for the trajectory plot")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Model helpers
# ---------------------------------------------------------------------------

def load_model(cfg, ckpt_path):
    model = instantiate(cfg.model)
    model.to(cfg.train.device)
    MyCheckpointer(model, save_dir=osp.dirname(ckpt_path)).resume_or_load(ckpt_path, resume=False)
    model.eval()
    return model


def run_inference(model, sample: dict) -> dict:
    """Run the model on a single (unbatched) PanoptesDataset sample."""
    inputs = {k: v.unsqueeze(0).cuda() for k, v in sample.items()}
    # rename 'choose' -> 'rgb_choose' (model key)
    if "choose" in inputs:
        inputs["rgb_choose"] = inputs.pop("choose")
    with torch.no_grad():
        end_points = model(inputs)
    return end_points


# ---------------------------------------------------------------------------
# Pose utilities
# ---------------------------------------------------------------------------

def make_T(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Build a 4x4 homogeneous transform from a (3,3) rotation and (3,) translation."""
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def load_gt_poses(dataset_path: str) -> list:
    """
    Load ground-truth poses for the relative navigation scenario.

    Relative navigation convention:
      - 'camera' is the chaser spacecraft (observer)
      - 'target' is the observed object
      - Both positions are given in ECI [metres]
      - Both rotations are body→ECI quaternions [x, y, z, w]

    Returns a list of dicts (sorted by 'id') each with:
      R_cam  : (3,3) rotation matrix, camera body → ECI
      p_cam  : (3,)  camera ECI position [m]
      R_tgt  : (3,3) rotation matrix, target body → ECI
      p_tgt  : (3,)  target ECI position [m]
    """
    labels_path = osp.join(dataset_path, "labels.json")
    with open(labels_path) as f:
        labels = json.load(f)
    labels = sorted(labels, key=lambda e: e["id"])

    poses = []
    for entry in labels:
        R_cam = Rotation.from_quat(entry["camera_rotation"]).as_matrix()  # body→ECI
        p_cam = np.array(entry["camera_eci_position"],  dtype=np.float64)
        R_tgt = Rotation.from_quat(entry["target_rotation"]).as_matrix()  # body→ECI
        p_tgt = np.array(entry["target_eci_position"],  dtype=np.float64)
        poses.append({"R_cam": R_cam, "p_cam": p_cam, "R_tgt": R_tgt, "p_tgt": p_tgt})
    return poses


def compute_gt_trajectory(gt_poses: list, indices: list) -> np.ndarray:
    """
    Compute the ground-truth relative-navigation trajectory using the
    same chaining convention as run_trajectory() so the two are comparable.

    The model predicts how a point P on the target transforms from the
    previous camera frame to the current one:
        P_cam_cur = R_rel @ P_cam_prev + t_rel

    The exact GT relative pose is obtained from
        T_tgt→cam_i  =  make_T( R_cam_i.T @ R_tgt_i,
                                 R_cam_i.T @ (p_tgt_i − p_cam_i) )
        T_rel  =  T_tgt→cam_cur  @  inv( T_tgt→cam_prev )

    which expands to:
        R_rel  =  R_cam_cur.T @ R_tgt_cur @ R_tgt_prev.T @ R_cam_prev
        t_rel  =  R_cam_cur.T @ (p_tgt_cur − p_cam_cur)
                  − R_rel @ R_cam_prev.T @ (p_tgt_prev − p_cam_prev)

    Accumulating from identity at the anchor frame gives a trajectory
    in the anchor's target-relative camera frame, starting at [0,0,0].

    Returns
    -------
    translations : (N, 3)  chained GT relative positions [metres]
    """
    T_global = np.eye(4)
    T_globals = [T_global.copy()]

    for k in range(1, len(indices)):
        prev = gt_poses[indices[k - 1]]
        cur  = gt_poses[indices[k]]

        # Pose of target body in each camera frame
        R_tc_prev = prev["R_cam"].T @ prev["R_tgt"]
        r_tc_prev = prev["R_cam"].T @ (prev["p_tgt"] - prev["p_cam"])   # [m]

        R_tc_cur  = cur["R_cam"].T  @ cur["R_tgt"]
        r_tc_cur  = cur["R_cam"].T  @ (cur["p_tgt"]  - cur["p_cam"])    # [m]

        # Step-by-step relative transform (same convention as model output)
        R_rel = R_tc_cur @ R_tc_prev.T
        t_rel = r_tc_cur - R_rel @ r_tc_prev

        T_rel    = make_T(R_rel, t_rel)
        T_global = T_global @ T_rel
        T_globals.append(T_global.copy())

    T_globals = np.stack(T_globals, axis=0)   # (N, 4, 4)
    return T_globals[:, :3, 3]                # (N, 3)


# ---------------------------------------------------------------------------
# Trajectory inference
# ---------------------------------------------------------------------------

def run_trajectory(model, dataset, start_idx: int, end_idx: int, template_offset: int = 5) -> dict:
    """
    Iterate over frames [start_idx, end_idx) stepping by `template_offset`,
    so each model call predicts the relative transform between frame `idx`
    and its template `idx - template_offset`.

    Chaining: T_global_i = T_global_{i-1} @ T_rel_i
    Anchor frame: start_idx - template_offset  (identity pose)

    Returns a dict with:
        poses        : (N, 4, 4)  global poses (N = number of frames including anchor)
        translations : (N, 3)    accumulated positions in anchor frame
        scores       : (N-1,)    per-step confidence scores
        indices      : (N,)      dataset frame indices (anchor first)
    """
    poses = [np.eye(4)]           # identity at anchor frame (start_idx - template_offset)
    scores = []
    indices = list(range(start_idx, end_idx, template_offset))

    for idx in tqdm(indices, desc="Trajectory inference"):
        sample = dataset[idx]
        ep = run_inference(model, sample)

        R_rel = ep["pred_R"].squeeze(0).cpu().numpy()   # (3, 3)
        t_rel = ep["pred_t"].squeeze(0).cpu().numpy()   # (3,)
        score = float(ep["pred_pose_score"].squeeze(0).cpu().numpy())

        T_rel = make_T(R_rel, t_rel)
        T_global = poses[-1] @ T_rel
        poses.append(T_global)
        scores.append(score)

    poses = np.stack(poses, axis=0)               # (N, 4, 4)
    translations = poses[:, :3, 3]               # (N, 3)
    all_indices = [start_idx - template_offset] + indices      # anchor frame first

    return dict(
        poses=poses,
        translations=translations,
        scores=np.array(scores),
        indices=all_indices,
    )


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_trajectory(result: dict, plot_path: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    txy = result["translations"]          # (N, 3) predicted
    scores = result["scores"]             # (N-1,)
    frame_ids = np.array(result["indices"])
    gt_txy = result.get("gt_translations")  # (N, 3) or None

    x, y, z = txy[:, 0], txy[:, 1], txy[:, 2]

    fig = plt.figure(figsize=(18, 10))
    fig.suptitle("UNOPose – Inferred vs GT Camera Trajectory (Panoptes)", fontsize=13)

    # --- 3D trajectory ---
    ax3d = fig.add_subplot(2, 3, (1, 4), projection="3d")
    ax3d.plot(x, y, z, linewidth=0.8, color="steelblue", alpha=0.8, label="predicted")
    ax3d.scatter(x[0], y[0], z[0], marker="^", s=80, color="green", zorder=5)
    ax3d.scatter(x[-1], y[-1], z[-1], marker="s", s=80, color="steelblue", zorder=5)
    if gt_txy is not None:
        gx, gy, gz = gt_txy[:, 0], gt_txy[:, 1], gt_txy[:, 2]
        ax3d.plot(gx, gy, gz, linewidth=0.8, color="tomato", alpha=0.8, label="GT")
        ax3d.scatter(gx[-1], gy[-1], gz[-1], marker="s", s=80, color="tomato", zorder=5)
    ax3d.set_xlabel("X [m]")
    ax3d.set_ylabel("Y [m]")
    ax3d.set_zlabel("Z [m]")
    ax3d.set_title("3D trajectory")
    ax3d.legend()

    # --- X, Y, Z vs frame index ---
    axis_labels = ["X [m]", "Y [m]", "Z [m]"]
    pred_colors = ["steelblue", "darkorange", "forestgreen"]
    subplot_positions = [2, 3, 5]          # skip position 4 (used by 3D span)
    for i, (lbl, col, pos) in enumerate(zip(axis_labels, pred_colors, subplot_positions)):
        ax = fig.add_subplot(2, 3, pos)
        ax.plot(frame_ids, txy[:, i], color=col, linewidth=1.2, label="predicted")
        if gt_txy is not None:
            ax.plot(frame_ids, gt_txy[:, i], color="tomato", linewidth=1.2,
                    linestyle="--", label="GT")
        ax.set_xlabel("frame index")
        ax.set_ylabel(lbl)
        ax.set_title(f"Translation {lbl[0]}")
        ax.legend(fontsize=8)
        ax.grid(True, linestyle="--", alpha=0.4)

    # --- confidence scores ---
    ax_sc = fig.add_subplot(2, 3, 6)
    ax_sc.plot(frame_ids[1:], scores, color="purple", linewidth=1.0)
    ax_sc.set_xlabel("frame index")
    ax_sc.set_ylabel("pred_pose_score")
    ax_sc.set_title("Confidence scores")
    ax_sc.grid(True, linestyle="--", alpha=0.4)

    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    logger.info(f"Trajectory plot saved to: {plot_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    if not osp.isfile(args.ckpt):
        logger.error(f"Checkpoint not found: {args.ckpt}")
        sys.exit(1)

    logger.info(f"Loading config from: {args.config_file}")
    cfg = LazyConfig.load(args.config_file)

    logger.info(f"Loading model from: {args.ckpt}")
    model = load_model(cfg, args.ckpt)
    logger.info("Model loaded and set to eval mode.")

    logger.info(f"Loading PanoptesDataset from: {args.dataset_path}")
    dataset = PanoptesDataset(args.dataset_path, template_offset=args.template_offset)
    fine_npoint = cfg.model.cfg.fine_npoint
    dataset.num_points = fine_npoint
    logger.info(f"Dataset size: {len(dataset)}, num_points set to {fine_npoint}")

    # ---------------------------------------------------------------- modes -
    if args.trajectory:
        K = args.template_offset
        start_idx = args.start_idx if args.start_idx is not None else K
        start_idx = max(start_idx, K)   # ensure anchor frame (start_idx - K) >= 0
        end_idx = args.end_idx if args.end_idx is not None else len(dataset)
        end_idx = min(end_idx, len(dataset))
        logger.info(f"Running trajectory inference for frames [{start_idx}, {end_idx}) step={K}")

        print(sum(p.numel() for p in model.parameters()))

        result = run_trajectory(model, dataset, start_idx, end_idx, template_offset=K)

        total_dist = float(np.sum(np.linalg.norm(np.diff(result["translations"], axis=0), axis=1)))
        logger.info(f"Processed {len(result['indices'])} frames, total path length: {total_dist:.4f} m")

        # Ground-truth trajectory from labels.json
        gt_poses = load_gt_poses(args.dataset_path)
        gt_translations = compute_gt_trajectory(gt_poses, result["indices"])
        result["gt_translations"] = gt_translations
        gt_dist = float(np.sum(np.linalg.norm(np.diff(gt_translations, axis=0), axis=1)))
        logger.info(f"GT total path length: {gt_dist:.4f} m")

        plot_trajectory(result, args.plot_path)

        npy_path = args.plot_path.replace(".png", "_poses.npy")
        np.save(npy_path, result["poses"])
        logger.info(f"Poses saved to: {npy_path}")

    else:
        # ------------------------------------------- single-sample inference
        idx = args.idx
        if idx < 0 or idx >= len(dataset):
            logger.error(f"Index {idx} out of range [0, {len(dataset) - 1}]")
            sys.exit(1)

        logger.info(f"Running single inference at index {idx} ...")
        sample = dataset[idx]
        ep = run_inference(model, sample)

        pred_R = ep["pred_R"].squeeze(0).cpu().numpy()
        pred_t = ep["pred_t"].squeeze(0).cpu().numpy()
        pred_score = float(ep["pred_pose_score"].squeeze(0).cpu().numpy())

        np.set_printoptions(precision=6, suppress=True)
        print(f"\n========== Inference result (idx={idx}) ==========")
        print(f"pred_R:\n{pred_R}")
        print(f"pred_t [meters]:  {pred_t}")
        print(f"pred_t [mm]:      {pred_t * 1000}")
        print(f"pred_pose_score:  {pred_score:.6f}")
        print("================================================\n")


if __name__ == "__main__":
    main()
