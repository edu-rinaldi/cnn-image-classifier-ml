from config import *
from utils import *

if __name__ == "__main__":
    
    model_name = 'LeNet_12Epochs_256x256_batch32'
    plot_history('history/'+model_name+'.json')


