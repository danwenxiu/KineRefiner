import sys
import argparse
import cv2
import os
import numpy as np
import torch
import torch.nn as nn
import glob
from tqdm import tqdm
import copy
from pathlib import Path

# -------------------------------------------------------------------------
# Project paths
# -------------------------------------------------------------------------
THIS_FILE = Path(__file__).resolve()
DEMO_DIR = THIS_FILE.parent
PROJECT_ROOT = DEMO_DIR.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(DEMO_DIR) not in sys.path:
    sys.path.insert(0, str(DEMO_DIR))

# -------------------------------------------------------------------------
# DBMambaPose/MHFormer demo dependencies
# Reuse demo/lib from the DBMambaPose repository.
# -------------------------------------------------------------------------
try:
    from lib.preprocess import h36m_coco_format
    from lib.hrnet.gen_kpts import gen_video_kpts as hrnet_pose
    from lib.utils import normalize_screen_coordinates, camera_to_world
except Exception as e:
    raise ImportError(
        "\n[ERROR] Demo dependencies were not found.\n"
        "Please copy the complete DBMambaPose 'demo/lib/' folder into:\n"
        f"    {DEMO_DIR / 'lib'}\n"
        "and make sure the YOLO/HRNet checkpoints required by that demo are present.\n"
        f"Original import error: {e}"
    ) from e

# -------------------------------------------------------------------------
# YOUR TCPFormer project imports
# These are intentionally the same loaders used by train.py.
# -------------------------------------------------------------------------
try:
    from utils.tools import get_config
    from utils.learning import load_model_TCPFormer
except Exception as e:
    raise ImportError(
        "\n[ERROR] Could not import TCPFormer model/config loaders.\n"
        "Run this script from the TCPFormer project root, e.g.:\n"
        "    python demo/vis_ours.py ...\n"
        f"Original import error: {e}"
    ) from e

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

plt.switch_backend("agg")
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42


# -------------------------------------------------------------------------
# Skeleton visualization
# -------------------------------------------------------------------------
def show2Dpose(kps, img):
    colors = [
        (171, 37, 36, 255),
        (39, 98, 53, 255),
        (43, 44, 124, 255),
    ]

    connections = [
        [0, 1], [1, 2], [2, 3],
        [0, 4], [4, 5], [5, 6],
        [0, 7], [7, 8], [8, 9], [9, 10],
        [8, 11], [11, 12], [12, 13],
        [8, 14], [14, 15], [15, 16],
    ]

    LR = [3, 3, 3, 3, 3, 3, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2]
    thickness = 3

    for j, c in enumerate(connections):
        start = list(map(int, kps[c[0]][:2]))
        end = list(map(int, kps[c[1]][:2]))
        color = colors[LR[j] - 1]

        cv2.line(img, (start[0], start[1]), (end[0], end[1]), color, thickness)
        cv2.circle(img, (start[0], start[1]), radius=3, color=color, thickness=-1)
        cv2.circle(img, (end[0], end[1]), radius=3, color=color, thickness=-1)

    return img


def show3Dpose(vals, ax):
    """Draw Human3.6M 17-joint skeleton."""
    ax.view_init(elev=15.0, azim=70)

    colors = [
        (43 / 255, 44 / 255, 124 / 255),
        (39 / 255, 98 / 255, 53 / 255),
        (171 / 255, 37 / 255, 36 / 255),
    ]

    I = np.array([0, 0, 1, 4, 2, 5, 0, 7, 8, 8, 14, 15, 11, 12, 8, 9])
    J = np.array([1, 4, 2, 5, 3, 6, 7, 8, 14, 11, 15, 16, 12, 13, 9, 10])
    LR = [3, 3, 3, 3, 3, 3, 1, 1, 2, 2, 2, 2, 2, 2, 1, 1]

    for i in np.arange(len(I)):
        x, y, z = [
            np.array([vals[I[i], j], vals[J[i], j]])
            for j in range(3)
        ]
        ax.plot(x, y, z, lw=3, color=colors[LR[i] - 1])

    radius = 0.72
    radius_z = 0.70
    xroot, yroot, zroot = vals[0, 0], vals[0, 1], vals[0, 2]

    ax.set_xlim3d([-radius + xroot, radius + xroot])
    ax.set_ylim3d([-radius + yroot, radius + yroot])
    ax.set_zlim3d([-radius_z + zroot, radius_z + zroot])
    ax.set_aspect("auto")

    white = (1.0, 1.0, 1.0, 0.0)
    ax.xaxis.set_pane_color(white)
    ax.yaxis.set_pane_color(white)
    ax.zaxis.set_pane_color(white)

    ax.tick_params("x", labelbottom=False)
    ax.tick_params("y", labelleft=False)
    ax.tick_params("z", labelleft=False)


def showimage(ax, img):
    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis("off")
    ax.imshow(img)


# -------------------------------------------------------------------------
# Video -> 2D keypoints
# -------------------------------------------------------------------------
def get_pose2D(video_path, output_dir):
    print("\n[1/3] Generating 2D pose...")

    keypoints, scores = hrnet_pose(
        video_path,
        det_dim=416,
        num_peroson=1,
        gen_output=True,
    )

    keypoints, scores, valid_frames = h36m_coco_format(keypoints, scores)

    # TCPFormer H36M config uses dim_in=3: x, y, confidence
    keypoints = np.concatenate((keypoints, scores[..., None]), axis=-1)

    output_dir_2d_input = os.path.join(output_dir, "input_2D")
    os.makedirs(output_dir_2d_input, exist_ok=True)

    output_npz = os.path.join(output_dir_2d_input, "keypoints.npz")
    np.savez_compressed(
        output_npz,
        reconstruction=keypoints,
        valid_frames=np.asarray(valid_frames),
    )

    print(f"[INFO] Saved 2D keypoints: {output_npz}")
    print(f"[INFO] 2D keypoint shape: {keypoints.shape}")


# -------------------------------------------------------------------------
# Clip preparation
# -------------------------------------------------------------------------
def resample(n_input_frames, target_frames):
    even = np.linspace(0, n_input_frames, num=target_frames, endpoint=False)
    result = np.floor(even)
    result = np.clip(
        result,
        a_min=0,
        a_max=max(n_input_frames - 1, 0),
    ).astype(np.uint32)
    return result


def turn_into_clips(keypoints, target_frames):
    """
    keypoints: [1, T, 17, 3]
    returns list of [1, target_frames, 17, 3]
    """
    clips = []
    clip_frame_indices = []
    n_frames = keypoints.shape[1]

    if n_frames <= target_frames:
        new_indices = resample(n_frames, target_frames)
        clips.append(keypoints[:, new_indices, ...])
        clip_frame_indices.append(new_indices)
    else:
        for start_idx in range(0, n_frames, target_frames):
            end_idx = min(start_idx + target_frames, n_frames)
            keypoints_clip = keypoints[:, start_idx:end_idx, ...]
            clip_length = keypoints_clip.shape[1]

            if clip_length != target_frames:
                local_indices = resample(clip_length, target_frames)
                clips.append(keypoints_clip[:, local_indices, ...])
                clip_frame_indices.append(start_idx + local_indices)
            else:
                clips.append(keypoints_clip)
                clip_frame_indices.append(
                    np.arange(start_idx, end_idx, dtype=np.int64)
                )

    return clips, clip_frame_indices


def flip_data(
    data,
    left_joints=[1, 2, 3, 14, 15, 16],
    right_joints=[4, 5, 6, 11, 12, 13],
):
    """
    data: [N, F, 17, D] or [F, 17, D]
    Works for numpy arrays and torch tensors.
    """
    if torch.is_tensor(data):
        flipped_data = data.clone()
    else:
        flipped_data = copy.deepcopy(data)

    flipped_data[..., 0] *= -1
    flipped_data[..., left_joints + right_joints, :] = \
        flipped_data[..., right_joints + left_joints, :]

    return flipped_data


# -------------------------------------------------------------------------
# Model loading
# -------------------------------------------------------------------------
def load_ours_model(opts, device):
    cfg = get_config(opts.config)

    print("\n[INFO] Building TCPFormer model using the SAME loader as train.py")
    print(f"[INFO] Config: {opts.config}")
    print(f"[INFO] n_frames from config: {cfg.n_frames}")
    print(f"[INFO] dim_in from config: {cfg.dim_in}")

    model = load_model_TCPFormer(cfg)

    # train.py wraps the model with DataParallel before loading the saved state_dict.
    if torch.cuda.is_available():
        model = nn.DataParallel(model)

    model = model.to(device)

    model_path = os.path.join(opts.checkpoint, opts.checkpoint_file)
    if not os.path.isfile(model_path):
        raise FileNotFoundError(
            f"\n[ERROR] Checkpoint not found:\n    {model_path}\n"
        )

    checkpoint = torch.load(model_path, map_location="cpu")

    if "model" not in checkpoint:
        raise KeyError(
            f"[ERROR] Checkpoint {model_path} does not contain key 'model'. "
            f"Available keys: {list(checkpoint.keys())}"
        )

    try:
        model.load_state_dict(checkpoint["model"], strict=True)
    except RuntimeError as e:
        # Helpful fallback if the checkpoint was saved without DataParallel.
        state = checkpoint["model"]
        has_module_prefix = any(k.startswith("module.") for k in state.keys())
        model_is_dp = isinstance(model, nn.DataParallel)

        if model_is_dp and not has_module_prefix:
            state = {f"module.{k}": v for k, v in state.items()}
            model.load_state_dict(state, strict=True)
        elif (not model_is_dp) and has_module_prefix:
            state = {k.replace("module.", "", 1): v for k, v in state.items()}
            model.load_state_dict(state, strict=True)
        else:
            raise e

    model.eval()

    print(f"[INFO] Loaded checkpoint: {model_path}")
    if "epoch" in checkpoint:
        print(f"[INFO] Checkpoint epoch: {checkpoint['epoch']}")
    if "min_mpjpe" in checkpoint:
        print(f"[INFO] Stored best MPJPE: {checkpoint['min_mpjpe']} mm")

    return model, cfg


# -------------------------------------------------------------------------
# 3D inference + visualization
# -------------------------------------------------------------------------
@torch.no_grad()
def get_pose3D(video_path, output_dir, opts):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, cfg = load_ours_model(opts, device)

    keypoint_file = os.path.join(output_dir, "input_2D", "keypoints.npz")
    data = np.load(keypoint_file, allow_pickle=True)
    keypoints = data["reconstruction"]

    target_frames = int(cfg.n_frames)
    clips, clip_frame_indices = turn_into_clips(keypoints, target_frames)

    cap = cv2.VideoCapture(video_path)
    video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if video_width <= 0 or video_height <= 0:
        raise RuntimeError(f"[ERROR] Failed to read video dimensions: {video_path}")

    vis_stride = max(int(opts.vis_stride), 1)

    # ---------------------------------------------------------------------
    # 2D pose frames
    # ---------------------------------------------------------------------
    print("\n[2/3] Generating 2D pose images...")
    output_dir_2d = os.path.join(output_dir, "pose2D")
    os.makedirs(output_dir_2d, exist_ok=True)

    i = 0
    while True:
        ret, img = cap.read()
        if not ret or img is None:
            break

        if i < keypoints.shape[1] and i % vis_stride == 0:
            input_2d = keypoints[0, i]
            image = show2Dpose(input_2d, copy.deepcopy(img))
            cv2.imwrite(
                os.path.join(output_dir_2d, f"{i:04d}_2D.png"),
                image,
            )
        i += 1

    cap.release()

    # ---------------------------------------------------------------------
    # 3D inference
    # ---------------------------------------------------------------------
    print("\n[3/3] Generating OUR 3D pose...")

    output_dir_3d = os.path.join(output_dir, "pose3D")
    os.makedirs(output_dir_3d, exist_ok=True)

    all_camera_preds = []
    all_world_preds = []
    all_original_frame_ids = []

    rot = np.array(
        [
            0.1407056450843811,
            -0.1500701755285263,
            -0.755240797996521,
            0.6223280429840088,
        ],
        dtype="float32",
    )

    for idx, (clip, original_ids) in enumerate(
        tqdm(list(zip(clips, clip_frame_indices)))
    ):
        # Keep the original DBMambaPose/MHFormer demo normalization.
        input_2d = normalize_screen_coordinates(
            clip,
            w=video_width,
            h=video_height,
        )
        input_2d_aug = flip_data(input_2d)

        input_2d = torch.from_numpy(
            input_2d.astype("float32")
        ).to(device)
        input_2d_aug = torch.from_numpy(
            input_2d_aug.astype("float32")
        ).to(device)

        output_3d_non_flip = model(input_2d)
        output_3d_flip = flip_data(model(input_2d_aug))
        output_3d = (output_3d_non_flip + output_3d_flip) / 2.0

        # Root-relative output, consistent with common H36M visualization.
        output_3d[:, :, 0, :] = 0

        camera_pred = output_3d[0].detach().cpu().numpy()

        # For a padded/resampled final clip, keep one prediction per
        # unique original video frame.
        _, unique_pos = np.unique(original_ids, return_index=True)
        unique_pos = np.sort(unique_pos)

        camera_pred = camera_pred[unique_pos]
        kept_original_ids = original_ids[unique_pos]

        world_pred_list = []

        for local_j, (post_out_cam, original_frame_id) in enumerate(
            zip(camera_pred, kept_original_ids)
        ):
            post_out_world = camera_to_world(
                post_out_cam.copy(),
                R=rot,
                t=0,
            )
            post_out_world[:, 2] -= np.min(post_out_world[:, 2])

            # Match the original demo visualization scale.
            max_abs = np.max(np.abs(post_out_world))
            if max_abs > 1e-8:
                post_out_vis = post_out_world / max_abs
            else:
                post_out_vis = post_out_world

            world_pred_list.append(post_out_world)

            if int(original_frame_id) % vis_stride != 0:
                continue

            fig = plt.figure(figsize=(9.6, 5.4))
            gs = gridspec.GridSpec(1, 1)
            gs.update(wspace=0.0, hspace=0.05)

            ax = plt.subplot(gs[0], projection="3d")
            show3Dpose(post_out_vis, ax)

            plt.savefig(
                os.path.join(
                    output_dir_3d,
                    f"{int(original_frame_id):04d}_3D.png",
                ),
                dpi=200,
                format="png",
                bbox_inches="tight",
            )
            plt.close(fig)

        all_camera_preds.append(camera_pred)
        all_world_preds.append(np.asarray(world_pred_list))
        all_original_frame_ids.append(np.asarray(kept_original_ids))

    if all_camera_preds:
        pred_camera = np.concatenate(all_camera_preds, axis=0)
        pred_world = np.concatenate(all_world_preds, axis=0)
        frame_ids = np.concatenate(all_original_frame_ids, axis=0)

        np.savez_compressed(
            os.path.join(output_dir, "pred_3d_camera.npz"),
            prediction=pred_camera,
            frame_ids=frame_ids,
        )
        np.savez_compressed(
            os.path.join(output_dir, "pred_3d_world.npz"),
            prediction=pred_world,
            frame_ids=frame_ids,
        )

        print(
            f"[INFO] Saved comparison-ready 3D predictions: "
            f"{pred_camera.shape}"
        )

    print("[INFO] Generating 3D pose successful!")

    # ---------------------------------------------------------------------
    # Compose Input | Ours panels
    # ---------------------------------------------------------------------
    image_2d_dir = sorted(
        glob.glob(os.path.join(output_dir_2d, "*_2D.png"))
    )
    image_3d_dir = sorted(
        glob.glob(os.path.join(output_dir_3d, "*_3D.png"))
    )

    map_2d = {
        Path(p).stem.split("_")[0]: p
        for p in image_2d_dir
    }
    map_3d = {
        Path(p).stem.split("_")[0]: p
        for p in image_3d_dir
    }
    common_ids = sorted(set(map_2d) & set(map_3d))

    output_dir_pose = os.path.join(output_dir, "pose")
    os.makedirs(output_dir_pose, exist_ok=True)

    print("\n[INFO] Composing Input | Ours panels...")

    for frame_id in tqdm(common_ids):
        image_2d = plt.imread(map_2d[frame_id])
        image_3d = plt.imread(map_3d[frame_id])

        # Crop as in the original DBMambaPose demo.
        if image_2d.shape[1] > image_2d.shape[0]:
            edge = (image_2d.shape[1] - image_2d.shape[0]) // 2
            if edge > 0:
                image_2d = image_2d[:, edge:image_2d.shape[1] - edge]

        edge = 130
        if image_3d.shape[0] > 2 * edge and image_3d.shape[1] > 2 * edge:
            image_3d = image_3d[
                edge:image_3d.shape[0] - edge,
                edge:image_3d.shape[1] - edge,
            ]

        font_size = 14
        fig = plt.figure(figsize=(15.0, 5.4))

        ax = plt.subplot(121)
        showimage(ax, image_2d)
        ax.set_title("Input", fontsize=font_size)

        ax = plt.subplot(122)
        showimage(ax, image_3d)
        ax.set_title("Ours", fontsize=font_size)

        plt.subplots_adjust(
            top=1,
            bottom=0,
            right=1,
            left=0,
            hspace=0,
            wspace=0,
        )
        plt.margins(0, 0)

        plt.savefig(
            os.path.join(output_dir_pose, f"{frame_id}_pose.png"),
            dpi=200,
            bbox_inches="tight",
        )
        plt.close(fig)


def img2video(video_path, output_dir):
    """Combine generated Input|Ours panels into an mp4."""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    if fps <= 0:
        fps = 25.0

    # We visualize every vis_stride frames; use a reasonable output FPS.
    output_fps = max(1.0, fps / max(1, GLOBAL_VIS_STRIDE))

    names = sorted(
        glob.glob(os.path.join(output_dir, "pose", "*.png"))
    )

    if not names:
        print("[WARN] No composed pose images found; skip video creation.")
        return

    first = cv2.imread(names[0])
    if first is None:
        print("[WARN] Could not read generated pose image; skip video.")
        return

    size = (first.shape[1], first.shape[0])
    video_name = Path(video_path).stem

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    output_video = os.path.join(output_dir, f"{video_name}_ours.mp4")

    writer = cv2.VideoWriter(
        output_video,
        fourcc,
        output_fps,
        size,
    )

    for name in names:
        img = cv2.imread(name)
        if img is not None:
            writer.write(img)

    writer.release()
    print(f"[INFO] Saved demo video: {output_video}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="In-the-wild visualization for our TCPFormer-based model."
    )

    parser.add_argument(
        "--video_dir",
        type=str,
        default="./demo/video/",
        help="Directory containing input .mp4/.avi/.mov videos.",
    )
    parser.add_argument(
        "--gpu",
        type=str,
        default="0",
        help="GPU device ID, e.g. 0.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/h36m/TCPFormer_h36m_243.yaml",
        help="Same YAML config used for training/evaluation.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoint_h36m_BPRE_HKCR",
        help="Checkpoint directory.",
    )
    parser.add_argument(
        "--checkpoint-file",
        type=str,
        default="best_epoch.pth.tr",
        help="Checkpoint filename.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./demo/output_ours/",
        help="Root output directory.",
    )
    parser.add_argument(
        "--vis_stride",
        type=int,
        default=9,
        help="Save one visualization every N original video frames.",
    )
    parser.add_argument(
        "--skip_2d",
        action="store_true",
        help="Reuse existing input_2D/keypoints.npz instead of running HRNet again.",
    )

    return parser.parse_args()


GLOBAL_VIS_STRIDE = 9


def main():
    global GLOBAL_VIS_STRIDE

    opts = parse_args()
    GLOBAL_VIS_STRIDE = max(int(opts.vis_stride), 1)

    # Set before the first CUDA model allocation.
    os.environ["CUDA_VISIBLE_DEVICES"] = opts.gpu

    video_dir = Path(opts.video_dir)
    if not video_dir.is_dir():
        raise NotADirectoryError(
            f"[ERROR] Video directory does not exist: {video_dir}"
        )

    video_extensions = {".mp4", ".avi", ".mov"}
    video_files = sorted(
        p for p in video_dir.iterdir()
        if p.suffix.lower() in video_extensions
    )

    if not video_files:
        raise FileNotFoundError(
            f"[ERROR] No .mp4/.avi/.mov files found in {video_dir}"
        )

    for video_path in video_files:
        video_name = video_path.stem
        output_dir = os.path.join(opts.output_dir, video_name)
        os.makedirs(output_dir, exist_ok=True)

        print("\n" + "=" * 72)
        print(f"[INFO] Processing video: {video_path}")
        print(f"[INFO] Output directory: {output_dir}")
        print("=" * 72)

        keypoint_file = os.path.join(
            output_dir,
            "input_2D",
            "keypoints.npz",
        )

        if opts.skip_2d and os.path.isfile(keypoint_file):
            print(f"[INFO] Reusing existing 2D keypoints: {keypoint_file}")
        else:
            get_pose2D(str(video_path), output_dir)

        get_pose3D(str(video_path), output_dir, opts)
        img2video(str(video_path), output_dir)

        print(f"[INFO] {video_name}: processed successfully.")

    print("\n[INFO] Generating demos for all videos successful!")


if __name__ == "__main__":
    main()