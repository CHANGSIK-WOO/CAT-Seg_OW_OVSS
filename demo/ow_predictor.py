# Copyright (c) Facebook, Inc. and its affiliates.
# Open-World Visualization Predictor for CAT-Seg
import atexit
import bisect
import multiprocessing as mp
from collections import deque
import json

import cv2
import torch
import numpy as np
from PIL import Image

from detectron2.data import MetadataCatalog
from detectron2.engine.defaults import DefaultPredictor
from detectron2.utils.video_visualizer import VideoVisualizer
from detectron2.utils.visualizer import ColorMode, Visualizer


class OWColorMode:
    """Extended color modes for Open-World visualization"""
    IMAGE = ColorMode.IMAGE
    SEGMENTATION = ColorMode.SEGMENTATION
    UNKNOWN_HIGHLIGHT = "unknown_highlight"  # Special mode for highlighting unknown regions


class OWVisualizer(Visualizer):
    """Extended Visualizer for Open-World semantic segmentation"""

    def __init__(self, img_rgb, metadata, scale=1.0, instance_mode=ColorMode.IMAGE, unknown_classes_start=75):
        super().__init__(img_rgb, metadata, scale, instance_mode)
        self.unknown_classes_start = unknown_classes_start

    def draw_sem_seg_with_unknown(self, sem_seg, area_threshold=None, alpha=0.8,
                                  show_class_names=True, show_unknown_regions=True):
        """Draw semantic segmentation with special handling for unknown classes."""
        if isinstance(sem_seg, torch.Tensor):
            sem_seg = sem_seg.numpy()

        labels, areas = np.unique(sem_seg, return_counts=True)
        sorted_idxs = np.argsort(-areas).tolist()
        labels = labels[sorted_idxs]

        binary_masks = []
        class_names = []
        colors = []

        for label in labels:
            # Filter by area if specified
            if area_threshold is not None and areas[label] < area_threshold:
                continue

            mask = (sem_seg == label)
            binary_masks.append(mask)

            # 🔧 수정: OWSS 모드에서 unknown 클래스 처리
            if label == self.unknown_classes_start:  # 정확히 75번만 unknown
                class_name = "UNKNOWN"
                if show_unknown_regions:
                    colors.append([1.0, 0.0, 0.0])  # Red for unknown
                else:
                    continue
            elif label < self.unknown_classes_start:  # 0~74: known classes
                # Known class names (first 75 classes from ADE150)
                if label < len(self.metadata.stuff_classes):
                    class_name = self.metadata.stuff_classes[label]
                else:
                    class_name = f"class_{label}"
                colors.append(None)  # Use default color
            else:  # > 75: invalid
                continue

            class_names.append(class_name)

        # Draw the segmentation
        for i, (mask, class_name) in enumerate(zip(binary_masks, class_names)):
            color = colors[i] if colors[i] is not None else None

            self.draw_binary_mask(
                mask,
                color=color,
                edge_color=None,
                text=class_name if show_class_names else None,
                alpha=alpha,
                area_threshold=area_threshold,
            )

        return self.output


class OWVisualizationDemo(object):
    def __init__(self, cfg, instance_mode=ColorMode.IMAGE, parallel=False):
        """
        Args:
            cfg (CfgNode): Configuration for the model
            instance_mode (ColorMode): Color mode for visualization
            parallel (bool): whether to run the model in different processes
        """
        self.metadata = MetadataCatalog.get(
            cfg.DATASETS.TEST[0] if len(cfg.DATASETS.TEST) else "__unused"
        )

        # Load class names from JSON if available
        self._load_class_mappings(cfg)

        self.cpu_device = torch.device("cpu")
        self.instance_mode = instance_mode
        self.unknown_classes_start = getattr(cfg.MODEL.SEM_SEG_HEAD, 'UNKNOWN_CLS', 75)

        self.parallel = parallel
        if parallel:
            num_gpu = torch.cuda.device_count()
            self.predictor = AsyncPredictor(cfg, num_gpus=num_gpu)
        else:
            self.predictor = DefaultPredictor(cfg)

        print(f"Demo initialized with unknown class threshold: {self.unknown_classes_start}")

    def _load_class_mappings(self, cfg):
        """Load class names from configuration"""
        try:
            # Try to load test class names
            if hasattr(cfg.MODEL.SEM_SEG_HEAD, 'TEST_CLASS_JSON'):
                test_json_path = cfg.MODEL.SEM_SEG_HEAD.TEST_CLASS_JSON
                if test_json_path and os.path.exists(test_json_path):
                    with open(test_json_path, 'r') as f:
                        test_classes = json.load(f)
                    print(f"Loaded {len(test_classes)} test class names from {test_json_path}")

                    # Update metadata with test classes if available
                    if hasattr(self.metadata, 'stuff_classes'):
                        # Extend or replace with test classes
                        extended_classes = test_classes + ["unknown"]
                        self.metadata.stuff_classes = extended_classes
        except Exception as e:
            print(f"Warning: Could not load class names: {e}")

    def run_on_image(self, image, enable_ow_mode=True, show_class_names=True,
                     show_unknown_regions=True, alpha=0.6):
        """
        Args:
            image (np.ndarray): an image of shape (H, W, C) (in BGR order)
            enable_ow_mode (bool): whether to enable open-world detection
            show_class_names (bool): whether to show class names on visualization
            show_unknown_regions (bool): whether to highlight unknown regions
            alpha (float): transparency for overlay

        Returns:
            predictions (dict): the output of the model
            vis_output (VisImage): the visualized image output
        """
        vis_output = None
        predictions = self.predictor(image)

        # Convert image from OpenCV BGR format to Matplotlib RGB format
        image = image[:, :, ::-1]

        # Create custom visualizer
        visualizer = OWVisualizer(
            image,
            self.metadata,
            instance_mode=self.instance_mode,
            unknown_classes_start=self.unknown_classes_start
        )

        if "sem_seg" in predictions:
            sem_seg = predictions["sem_seg"].argmax(dim=0).to(self.cpu_device)

            # Log prediction statistics
            unique_classes = torch.unique(sem_seg)
            known_classes = unique_classes[unique_classes < self.unknown_classes_start]
            unknown_classes = unique_classes[unique_classes >= self.unknown_classes_start]

            print(f"Prediction stats - Known classes: {len(known_classes)}, Unknown classes: {len(unknown_classes)}")
            if len(unknown_classes) > 0:
                print(f"Unknown class IDs: {unknown_classes.tolist()}")

            # Use custom visualization method
            vis_output = visualizer.draw_sem_seg_with_unknown(
                sem_seg,
                alpha=alpha,
                show_class_names=show_class_names,
                show_unknown_regions=show_unknown_regions
            )

        elif "instances" in predictions:
            instances = predictions["instances"].to(self.cpu_device)
            vis_output = visualizer.draw_instance_predictions(predictions=instances)
        else:
            print("Warning: No segmentation predictions found")
            vis_output = visualizer.output

        return predictions, vis_output

    def _frame_from_video(self, video):
        while video.isOpened():
            success, frame = video.read()
            if success:
                yield frame
            else:
                break

    def run_on_video(self, video, enable_ow_mode=True, show_class_names=True,
                     show_unknown_regions=True, alpha=0.6):
        """
        Visualizes predictions on frames of the input video.

        Args:
            video (cv2.VideoCapture): a :class:`VideoCapture` object
            enable_ow_mode (bool): whether to enable open-world detection
            show_class_names (bool): whether to show class names
            show_unknown_regions (bool): whether to highlight unknown regions
            alpha (float): transparency for overlay

        Yields:
            ndarray: BGR visualizations of each video frame
        """
        video_visualizer = VideoVisualizer(self.metadata, self.instance_mode)

        def process_predictions(frame, predictions):
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            if "sem_seg" in predictions:
                # Use our custom visualization for video too
                visualizer = OWVisualizer(
                    frame,
                    self.metadata,
                    instance_mode=self.instance_mode,
                    unknown_classes_start=self.unknown_classes_start
                )

                sem_seg = predictions["sem_seg"].argmax(dim=0).to(self.cpu_device)
                vis_frame = visualizer.draw_sem_seg_with_unknown(
                    sem_seg,
                    alpha=alpha,
                    show_class_names=show_class_names,
                    show_unknown_regions=show_unknown_regions
                )
                vis_frame = vis_frame.get_image()

            elif "instances" in predictions:
                predictions_cpu = predictions["instances"].to(self.cpu_device)
                vis_frame = video_visualizer.draw_instance_predictions(frame, predictions_cpu)
                vis_frame = vis_frame.get_image()
            else:
                vis_frame = frame

            # Converts Matplotlib RGB format to OpenCV BGR format
            vis_frame = cv2.cvtColor(vis_frame, cv2.COLOR_RGB2BGR)
            return vis_frame

        frame_gen = self._frame_from_video(video)
        if self.parallel:
            buffer_size = self.predictor.default_buffer_size

            frame_data = deque()

            for cnt, frame in enumerate(frame_gen):
                frame_data.append(frame)
                self.predictor.put(frame)

                if cnt >= buffer_size:
                    frame = frame_data.popleft()
                    predictions = self.predictor.get()
                    yield process_predictions(frame, predictions)

            while len(frame_data):
                frame = frame_data.popleft()
                predictions = self.predictor.get()
                yield process_predictions(frame, predictions)
        else:
            for frame in frame_gen:
                predictions = self.predictor(frame)
                yield process_predictions(frame, predictions)


class AsyncPredictor:
    """
    A predictor that runs the model asynchronously, possibly on >1 GPUs.
    """

    class _StopToken:
        pass

    class _PredictWorker(mp.Process):
        def __init__(self, cfg, task_queue, result_queue):
            self.cfg = cfg
            self.task_queue = task_queue
            self.result_queue = result_queue
            super().__init__()

        def run(self):
            predictor = DefaultPredictor(self.cfg)

            while True:
                task = self.task_queue.get()
                if isinstance(task, AsyncPredictor._StopToken):
                    break
                idx, data = task
                result = predictor(data)
                self.result_queue.put((idx, result))

    def __init__(self, cfg, num_gpus: int = 1):
        """
        Args:
            cfg (CfgNode):
            num_gpus (int): if 0, will run on CPU
        """
        num_workers = max(num_gpus, 1)
        self.task_queue = mp.Queue(maxsize=num_workers * 3)
        self.result_queue = mp.Queue(maxsize=num_workers * 3)
        self.procs = []
        for gpuid in range(max(num_gpus, 1)):
            cfg = cfg.clone()
            cfg.defrost()
            cfg.MODEL.DEVICE = "cuda:{}".format(gpuid) if num_gpus > 0 else "cpu"
            self.procs.append(
                AsyncPredictor._PredictWorker(cfg, self.task_queue, self.result_queue)
            )

        self.put_idx = 0
        self.get_idx = 0
        self.result_rank = []
        self.result_data = []

        for p in self.procs:
            p.start()
        atexit.register(self.shutdown)

    def put(self, image):
        self.put_idx += 1
        self.task_queue.put((self.put_idx, image))

    def get(self):
        self.get_idx += 1  # the index needed for this request
        if len(self.result_rank) and self.result_rank[0] == self.get_idx:
            res = self.result_data[0]
            del self.result_data[0], self.result_rank[0]
            return res

        while True:
            # make sure the results are returned in the correct order
            idx, res = self.result_queue.get()
            if idx == self.get_idx:
                return res
            insert = bisect.bisect(self.result_rank, idx)
            self.result_rank.insert(insert, idx)
            self.result_data.insert(insert, res)

    def __len__(self):
        return self.put_idx - self.get_idx

    def __call__(self, image):
        self.put(image)
        return self.get()

    def shutdown(self):
        for _ in self.procs:
            self.task_queue.put(AsyncPredictor._StopToken())

    @property
    def default_buffer_size(self):
        return len(self.procs) * 5