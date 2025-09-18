# Copyright (c) Facebook, Inc. and its affiliates.
# Modified for Open-World CAT-Seg Demo
import argparse
import glob
import multiprocessing as mp
import os
import sys
import tempfile
import time
import warnings

import cv2
import numpy as np
import tqdm

from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.utils.logger import setup_logger

from cat_seg import add_cat_seg_config
from ow_predictor import OWVisualizationDemo

# constants
WINDOW_NAME = "Open-World CAT-Seg Demo"


def setup_cfg(args):
    """Load config from file and command-line arguments"""
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_cat_seg_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    return cfg


def get_parser():
    parser = argparse.ArgumentParser(description="Open-World CAT-Seg Demo")
    parser.add_argument(
        "--config-file",
        default="configs/ow_vitb_384_size_down.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument("--webcam", action="store_true", help="Take inputs from webcam.")
    parser.add_argument("--video-input", help="Path to video file.")
    parser.add_argument(
        "--input",
        nargs="+",
        help="A list of space separated input images; "
             "or a single glob pattern such as 'directory/*.jpg'",
    )
    parser.add_argument(
        "--output",
        help="A file or directory to save output visualizations. "
             "If not given, will show output in an OpenCV window.",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--model-weights",
        type=str,
        required=True,
        help="Path to model weights (.pth file)",
    )
    parser.add_argument(
        "--enable-ow-mode",
        action="store_true",
        default=True,
        help="Enable Open-World mode for unknown class detection",
    )
    parser.add_argument(
        "--show-class-names",
        action="store_true",
        default=True,
        help="Show class names on visualization",
    )
    parser.add_argument(
        "--show-unknown-regions",
        action="store_true",
        default=True,
        help="Highlight unknown regions in the visualization",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.6,
        help="Transparency for semantic segmentation overlay (0.0-1.0)",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    return parser


def test_opencv_video_format(codec, file_ext):
    with tempfile.TemporaryDirectory(prefix="video_format_test") as dir:
        filename = os.path.join(dir, "test_file" + file_ext)
        writer = cv2.VideoWriter(
            filename=filename,
            fourcc=cv2.VideoWriter_fourcc(*codec),
            fps=float(30),
            frameSize=(10, 10),
            isColor=True,
        )
        [writer.write(np.zeros((10, 10, 3), np.uint8)) for _ in range(30)]
        writer.release()
        if os.path.isfile(filename):
            return True
        return False


def process_image(demo, path, args, logger):
    """Process a single image and return results"""
    # Read image
    img = read_image(path, format="BGR")
    start_time = time.time()

    # Run inference
    predictions, visualized_output = demo.run_on_image(
        img,
        enable_ow_mode=args.enable_ow_mode,
        show_class_names=args.show_class_names,
        show_unknown_regions=args.show_unknown_regions,
        alpha=args.alpha
    )

    # Log timing and results
    inference_time = time.time() - start_time

    # Count predicted classes
    if "sem_seg" in predictions:
        pred_classes = np.unique(predictions["sem_seg"].argmax(dim=0).cpu().numpy())
        num_classes = len(pred_classes)
        known_classes = len([c for c in pred_classes if c < 75])  # Assuming first 75 are known
        unknown_classes = num_classes - known_classes

        logger.info(
            f"{path}: Processed in {inference_time:.2f}s - "
            f"Total classes: {num_classes}, Known: {known_classes}, Unknown: {unknown_classes}"
        )
    else:
        logger.info(f"{path}: Processed in {inference_time:.2f}s")

    return predictions, visualized_output


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    args = get_parser().parse_args()
    setup_logger(name="fvcore")
    logger = setup_logger()
    logger.info("Arguments: " + str(args))

    # Setup configuration
    cfg = setup_cfg(args)

    # Override model weights
    cfg.defrost()
    cfg.MODEL.WEIGHTS = args.model_weights
    if args.enable_ow_mode:
        cfg.MODEL.SEM_SEG_HEAD.ENABLE_OW_MODE = True
    cfg.freeze()

    # Create demo object
    demo = OWVisualizationDemo(cfg)

    # Process inputs
    if args.input:
        if len(args.input) == 1:
            args.input = glob.glob(os.path.expanduser(args.input[0]))
            assert args.input, "The input path(s) was not found"

        for path in tqdm.tqdm(args.input, disable=not args.output):
            predictions, visualized_output = process_image(demo, path, args, logger)

            if args.output:
                if os.path.isdir(args.output):
                    assert os.path.isdir(args.output), args.output
                    out_filename = os.path.join(args.output, os.path.basename(path))
                else:
                    assert len(args.input) == 1, "Please specify a directory with args.output"
                    out_filename = args.output
                visualized_output.save(out_filename)
                logger.info(f"Saved visualization to {out_filename}")
            else:
                cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
                cv2.imshow(WINDOW_NAME, visualized_output.get_image()[:, :, ::-1])
                key = cv2.waitKey(0)
                if key == 27:  # ESC key
                    break
                elif key == ord('s'):  # Save current image
                    save_path = f"demo_output_{os.path.basename(path)}"
                    visualized_output.save(save_path)
                    logger.info(f"Saved current image to {save_path}")

    elif args.webcam:
        assert args.input is None, "Cannot have both --input and --webcam!"
        assert args.output is None, "output not yet supported with --webcam!"
        cam = cv2.VideoCapture(0)

        logger.info("Starting webcam demo. Press 'q' to quit, 's' to save current frame")
        frame_count = 0

        while True:
            ret, frame = cam.read()
            if not ret:
                break

            frame_count += 1

            # Process every few frames to maintain real-time performance
            if frame_count % 5 == 0:  # Process every 5th frame
                predictions, vis_frame = demo.run_on_image(
                    frame,
                    enable_ow_mode=args.enable_ow_mode,
                    show_class_names=args.show_class_names,
                    show_unknown_regions=args.show_unknown_regions,
                    alpha=args.alpha
                )
                display_frame = vis_frame.get_image()[:, :, ::-1]
            else:
                display_frame = frame

            cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
            cv2.imshow(WINDOW_NAME, display_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s') and frame_count % 5 == 0:
                save_path = f"webcam_frame_{frame_count}.jpg"
                vis_frame.save(save_path)
                logger.info(f"Saved frame to {save_path}")

        cam.release()
        cv2.destroyAllWindows()

    elif args.video_input:
        video = cv2.VideoCapture(args.video_input)
        width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frames_per_second = video.get(cv2.CAP_PROP_FPS)
        num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        basename = os.path.basename(args.video_input)

        codec, file_ext = (
            ("x264", ".mkv") if test_opencv_video_format("x264", ".mkv") else ("mp4v", ".mp4")
        )
        if codec == "mp4v":
            warnings.warn("x264 codec not available, switching to mp4v")

        if args.output:
            if os.path.isdir(args.output):
                output_fname = os.path.join(args.output, basename)
                output_fname = os.path.splitext(output_fname)[0] + file_ext
            else:
                output_fname = args.output
            assert not os.path.isfile(output_fname), output_fname
            output_file = cv2.VideoWriter(
                filename=output_fname,
                fourcc=cv2.VideoWriter_fourcc(*codec),
                fps=float(frames_per_second),
                frameSize=(width, height),
                isColor=True,
            )

        assert os.path.isfile(args.video_input)

        for frame_idx in tqdm.tqdm(range(num_frames)):
            ret, frame = video.read()
            if not ret:
                break

            predictions, vis_frame = demo.run_on_image(
                frame,
                enable_ow_mode=args.enable_ow_mode,
                show_class_names=args.show_class_names,
                show_unknown_regions=args.show_unknown_regions,
                alpha=args.alpha
            )
            vis_frame_bgr = vis_frame.get_image()[:, :, ::-1]

            if args.output:
                output_file.write(vis_frame_bgr)
            else:
                cv2.namedWindow(basename, cv2.WINDOW_NORMAL)
                cv2.imshow(basename, vis_frame_bgr)
                if cv2.waitKey(1) == 27:
                    break

        video.release()
        if args.output:
            output_file.release()
            logger.info(f"Saved video to {output_fname}")
        else:
            cv2.destroyAllWindows()

    logger.info("Demo completed!")