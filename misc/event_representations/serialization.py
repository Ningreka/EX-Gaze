import h5py
import numpy as np


class Save2hdf5:
    def __init__(self, file_path,compression_opts=4, **kwargs):
        self.file = h5py.File(file_path, 'w')
        self.compression_opts = compression_opts
        for k,v in kwargs.items():
            self.file.attrs[k] = v

    def save(self, key, event_volume):
        self.file.create_dataset(name=key, data=np.asfarray(event_volume, dtype=np.float32),
                                 compression="gzip",compression_opts=self.compression_opts)
    def is_exists(self, key):
        return key in self.file
    def close(self):
        self.file.close()

    def __del__(self):
        self.file.close()


class Save2Memory:
    def __init__(self):
        self.volumes = dict()

    def save(self, key, event_volume):
        self.volumes[key] = event_volume

    def is_exists(self, key):
        return key in self.volumes

