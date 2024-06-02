import random
import numpy as np
import tensorflow as tf

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

if __name__ == '__main__':
    SEED = 1601
    set_seed(SEED)
