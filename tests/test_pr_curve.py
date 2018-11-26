import torch
import numpy as np
from tensorboardX import SummaryWriter

true_positive_counts = [75, 64, 21, 5, 0]
false_positive_counts = [150, 105, 18, 0, 0]
true_negative_counts = [0, 45, 132, 150, 150]
false_negative_counts = [0, 11, 54, 70, 75]
precision = [0.3333333, 0.3786982, 0.5384616, 1.0, 0.0]
recall = [1.0, 0.8533334, 0.28, 0.0666667, 0.0]

with SummaryWriter() as writer:
    writer.add_pr_curve('xoxo', np.random.randint(2, size=100), np.random.rand(
        100), 1)  # needs tensorboard 0.4RC or later
    writer.add_pr_curve_raw('prcurve with raw data', true_positive_counts,
                            false_positive_counts,
                            true_negative_counts,
                            false_negative_counts,
                            precision,
                            recall, 1)