import numpy as np

class DataIterator:

    def __init__(self,
                 data: dict,
                 start_index: int = 0):
        self.data = data
        self.keys = list(self.data.keys())
        self.index = start_index
        super().__init__()
    
    def next(self) -> tuple[list, list]:
            
        output = []

        for key in self.keys:
            try:
                output.append(self.data[key][self.index])
            except IndexError:
                output.append(self.data[key][0])

        self.index += 1

        return (output, self.keys)