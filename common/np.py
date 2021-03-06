from common.config import GPU, GPU_ID


if GPU:
    from chainer import cuda
    device = GPU_ID
    cuda.get_device(device).use()
    import cupy as np
    np.cuda.set_allocator(np.cuda.MemoryPool().malloc)
    np.add.at = np.scatter_add

    print('\033[92m' + '-' * 60 + '\033[0m')
    print(' ' * 23 + '\033[92mGPU Mode (cupy)\033[0m')
    print('\033[92m' + '-' * 60 + '\033[0m\n')

else:
    import numpy as np
