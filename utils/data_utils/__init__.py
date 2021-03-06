from utils.data_utils.power import POWER
from utils.data_utils.gas import GAS
from utils.data_utils.hepmass import HEPMASS
from utils.data_utils.miniboone import MINIBOONE
from utils.data_utils.bsds300 import BSDS300
from utils.data_utils.bsds300 import BSDS300
from utils.data_utils.cifar10 import CIFAR10
from utils.data_utils.mnist import MNIST

data_mapping = {'BSDS300': BSDS300,
                'GAS': GAS,
                'MINIBOONE': MINIBOONE,
                'POWER': POWER,
                'HEPMASS': HEPMASS
                }
