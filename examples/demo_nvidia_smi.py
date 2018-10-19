"""
write gpu and (gpu) memory usage of nvidia cards as scalar
"""
from tensorboardX import SummaryWriter
import time

# package: nvidia-ml-py or nvidia-ml-py3
try:
    import nvidia_smi
except ImportError:
    nvidia_smi = None


with SummaryWriter() as writer:

    # only run when nvidia_smi is installed
    if nvidia_smi:
        nvidia_smi.nvmlInit()
        handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)  # gpu0

        for n_iter in range(10):
            res = nvidia_smi.nvmlDeviceGetUtilizationRates(handle)
            writer.add_scalar('nv/gpu', res.gpu, n_iter)
            writer.add_scalar('nv/gpu_mem', res.memory, n_iter)
            time.sleep(0.1)
