import os
DATAPATH = os.path.join(os.path.abspath('.'), 'pmodel_data','job')
NODE_DIM = 1000
class arguments():
    def __init__(self):
        self.cuda = True
        self.fastmode = False
        self.seed = 42
        self.epochs = 200
        self.lr = 0.01
        self.weight_decay = 5e-4
        self.hidden = 16
        self.dropout = 0.5
args = arguments()