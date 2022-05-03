import argparse
import os
import sys

from pyanitools import anidataloader
import h5py

from sklearn.model_selection import train_test_split

from torch import nn

import train_test as tt

if __name__ == "__main__":
    # Define an argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data', 
                        required=True, 
                        dest='data_files', 
                        help='Paths of all h5 files to be loaded in, '\
                             'separated by commas. E.g. '\
                             '-d ../ANI-1_release/ani_gdb_s03.h5,'\
                             '../ANI-1_release/ani_gdb_s04.h5,'\
                             '../ANI-1_release/ani_gdb_s05.h5', 
                        type=str)
    parser.add_argument('-c', '--checkpoint', 
                        required=False, 
                        dest='cp_file', 
                        help='File containing the checkpoint '\
                             'for the last epoch', 
                        type=str, 
                        default='checkpoint.pt')
    parser.add_argument('-v', '--validation_size', 
                        required=False, 
                        dest='val_size', 
                        help='Decimal percentage for size of validation set '\
                             'by number of molecules. '\
                             'Must be in the range [0.0, 1.0). '\
                             'E.g. -t 0.2 means 80 percent of the molecules '\
                             'are used for training, '\
                             'and 20 percent for testing', 
                        type=float, 
                        default=0.2)
    parser.add_argument('-e', '--epochs', 
                        required=False, 
                        dest ='max_epochs', 
                        help='Maximum number of epochs', 
                        type=int, 
                        default=100)
    parser.add_argument('-b', '--batch-size', 
                        required=False, 
                        dest ='batch_size', 
                        help='Size of each batch', 
                        type=int, 
                        default=1024)
    parser.add_argument('--use-gpu', 
                        required=False, 
                        dest ='use_gpu', 
                        help='Toggle GPU usage to True', 
                        action="store_true")
    args = parser.parse_args()

    # definte local variables inputted into argument parser
    checkpoint_file = args.cp_file
    epochs = args.max_epochs
    batchsize = args.batch_size
    to_use_gpu = args.use_gpu

    # assert that a portion of the data will be left out as validation
    val_size = args.val_size
    if val_size < 0 or val_size >= 1:
        sys.exit("Please enter a test size value in the range [0.0, 1.0).")

    data_files = args.data_files

    file_L = data_files.strip().split(',')

    data = []

    # extract data from h5 file into a data list
    print("Loading data files...")
    
    for filename in file_L:
        if not os.path.isfile(filename):
            sys.exit(filename + " does not exist.")

        data = data + list(anidataloader(filename))

        print(filename + " successfully loaded.")

    print("...data files loaded.")

    data_list = [item for item in data]

    # training the data
    print("Loading/Saving model to: " + checkpoint_file)
    print("Training until: " + str(epochs) + \
          " epochs have been completed with batch size: " + str(batchsize))
    print("Validating with validation size of " + str(val_size))
    print("GPU usage set to: "  + str(to_use_gpu))

    print("Beginning training...")

    
    arch_dict = {'C': nn.Sequential(
                      nn.Linear(384, 144),
                      nn.CELU(),
                      nn.Linear(144,112),
                      nn.CELU(),
                      nn.Linear(112,96),
                      nn.CELU(),
                      nn.Linear(96,1)), 
                 'H': nn.Sequential(
                      nn.Linear(384, 160),
                      nn.CELU(),
                      nn.Linear(160,128),
                      nn.CELU(),
                      nn.Linear(128,96),
                      nn.CELU(),
                      nn.Linear(96,1)),
                 'O': nn.Sequential(
                      nn.Linear(384, 128),
                      nn.CELU(),
                      nn.Linear(128,112),
                      nn.CELU(),
                      nn.Linear(112,96),
                      nn.CELU(),
                      nn.Linear(96,1)),
                 'N': nn.Sequential(
                      nn.Linear(384, 128),
                      nn.CELU(),
                      nn.Linear(128,112),
                      nn.CELU(),
                      nn.Linear(112,96),
                      nn.CELU(),
                      nn.Linear(96,1))}

    epoch_losses, val_losses = \
        tt.train(data_list, arch_dict, checkpoint_file, 
                 max_epochs=epochs, 
                 validation_size=val_size, 
                 batch_size=batchsize, 
                 use_gpu=to_use_gpu)

    print("...training complete!")
    





