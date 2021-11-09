import numpy as np
from tensorflow.keras import utils

## Used for data Generation
class DataGenerator(utils.Sequence):
    def __init__(self,data,label,way,batch_size=32,shuffle=False):
        super().__init__()
        ## Create empty slot
        self.dataset=np.empty((list(data.shape)+[1]),dtype=np.float32)
        # Import data in array format
        self.dataset[...,0]=data
        # One hot encoding
        self.labels=utils.to_categorical(label,way)
        # record parameters
        self.batch_size=batch_size
        self.shuffle=shuffle
        
    def __len__(self):
        return len(self.dataset)//self.batch_size
    def __getitem__(self,index):
        # preprocessing: only normalization
        x=self.dataset[index*self.batch_size:(index+1)*self.batch_size]/255.
        y=self.labels[index*self.batch_size:(index+1)*self.batch_size]
        return x,y
    def __iter__(self):
        """Create a generator that iterate over the Sequence."""
        # Suffle is only needed during training 
        if self.shuffle:
            # re-order the data series
            order=np.random.permutation(len(self.dataset))
            # re-arrange data by the shuffled order
            self.dataset=self.dataset[order]
            self.labels=self.labels[order]
        # output the data with the default order
        for item in (self[i] for i in range(self.__len__())):
            yield item
