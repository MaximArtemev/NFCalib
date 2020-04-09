import os
from utils import definitions

if __name__ == '__main__':
    os.system('wget -O processed_data.tar.gz https://zenodo.org/record/1161203/files/data.tar.gz?download=1')
    os.system(f'tar -xvf processed_data.tar.gz -C {definitions.DATA_DIR} --strip-components=1')
    os.system('rm processed_data.tar.gz')


