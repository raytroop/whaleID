import logging
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def set_logger(log_name, log_path):
    """Set the logger to log info in terminal and file `log_path`.

    In general, it is useful to have a logger so that every output to the terminal is saved
    in a permanent file. Here we save it to `model_dir/train.log`.

    Args:
        log_name: (string) name of log,  if name is None, return a logger which is the root logger of the hierarchy
        log_path: (string) where to log
    """

    logger = logging.getLogger(log_name)
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        # Logging to a file
        file_handler = logging.FileHandler(log_path, mode='w+')
        file_handler.setFormatter(logging.Formatter(
            '%(asctime)s:%(levelname)s: %(message)s'))
        logger.addHandler(file_handler)

        # Logging to console
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(
            logging.Formatter('%(levelname)s: %(message)s'))
        logger.addHandler(stream_handler)
    return logger

class RunningAverage():
    """A simple class that maintains the running average of a quantity

    Example:
    ```
    loss_avg = RunningAverage()
    loss_avg.update(2)
    loss_avg.update(4)
    loss_avg() = 3
    ```
    """

    def __init__(self):
        self.steps = 0
        self.total = 0.0

    def update(self, val):
        self.total += val
        self.steps += 1

    def __call__(self):
        return self.total/float(self.steps)


def iou_sim(box1, box2):
    box1Area = np.maximum((box1[:, 2] - box1[:, 0]), 0.0) * \
        np.maximum((box1[:, 3] - box1[:, 1]), 0.0)
    box2Area = np.maximum((box2[:, 2] - box2[:, 0]), 0.0) * \
        np.maximum((box2[:, 3] - box2[:, 1]), 0.0)
    x0 = np.maximum(box1[:, 0], box2[:, 0])
    y0 = np.maximum(box1[:, 1], box2[:, 1])
    x1 = np.minimum(box1[:, 2], box2[:, 2])
    y1 = np.minimum(box1[:, 3], box2[:, 3])
    interArea = np.maximum(x1 - x0, 0.0) * np.maximum(y1 - y0, 0.0)

    iou = interArea / (box1Area + box2Area - interArea)

    return np.mean(iou)


def draw_bbox(img, predbox, gtbox=None):
    # Create figure and axes
    _, ax = plt.subplots(1)
    # Display the image
    ax.imshow(img, cmap='gray')
    rect = patches.Rectangle(predbox[:2], predbox[2]-predbox[0],
                             predbox[3]-predbox[1], linewidth=2, edgecolor='b', facecolor='none')
    # Add the patch to the Axes
    ax.add_patch(rect)
    if gtbox is not None:
        # Create a Rectangle patch
        rect = patches.Rectangle(
            gtbox[:2], gtbox[2]-gtbox[0], gtbox[3]-gtbox[1], linewidth=2, edgecolor='r', facecolor='none')
        # Add the patch to the Axes
        ax.add_patch(rect)

    plt.show()
