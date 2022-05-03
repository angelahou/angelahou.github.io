import argparse
import os
import sys
import pickle

from pyanitools import anidataloader
import h5py

from sklearn.model_selection import train_test_split

from torch import nn

import train_test as tt

def write_output(results, output_name):
    """
    Writes the predicted losses to a file.

    Inputs:
    -------
    results : dict
        A dictionary containing the predicted molecular energies 
        and losses of the testing set
    output_name : str
        The file name to write to.
    """
    # create losses file
    losses_csv_file = output_name + "_losses.csv"
    print("Writing losses to: " + losses_csv_file)
    
    # open a file stream to write in losses
    with open(losses_csv_file, 'w') as of:
        of.write("Losses")
        of.write('\n')
        for loss in results['Test Losses']:
            of.write(str(loss))
            of.write('\n')
    print("Losses file written.")
    
    # open a file to pickle energies
    energies_csv_file = output_name + "_energies.pickle"
    print("Pickling energies to: " + losses_csv_file)
    
    with open(energies_csv_file, 'wb') as of:
        pickle.dump(results['Molecular Energies'], of)
        
    print("Energies pickled.")

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
    parser.add_argument('-o', '--output', 
                        required=False, 
                        dest='o_file', 
                        help='Prefix of output files to be written.', 
                        type=str, 
                        default="output")
    parser.add_argument('-c', '--checkpoint', 
                        required=False, 
                        dest='cp_file', 
                        help='File containing the checkpoint '\
                             'for the trained model', 
                        type=str, 
                        default='checkpoint.pt')
    parser.add_argument('--use-gpu', 
                        required=False, 
                        dest ='use_gpu', 
                        help='Toggle GPU usage to True', 
                        action="store_true")
    args = parser.parse_args()

    # create a checkpoint file 
    # to save training status if we run out of time on GPUs
    checkpoint_file = args.cp_file
    # flag for whether to use Cori GPUs or local machine
    to_use_gpu = args.use_gpu
    # output file to write to
    output_file = args.o_file
    # h5 files from the ANI-1 dataset
    data_files = args.data_files
    # list of data files
    file_L = data_files.strip().split(',')

    data = []

    # Load in all data files from file list
    print("Loading data files...")
    
    for filename in file_L:
        if not os.path.isfile(filename):
            sys.exit(filename + " does not exist.")

        data = data + list(anidataloader(filename))

        print(filename + " successfully loaded.")

    print("...data files loaded.")

    # A list of dictionaries, where each item refers to a molecule 
    # and stores information about its conformations and related energies
    data_list = [item for item in data]
    
    # create a dictionary of NN architectures, to pass into the subnet module. 
    # Each key is an atom type, and its value is the architecture
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

    print("Beginning testing...")
    test_results = \
        tt.predict_energies(data_list, arch_dict, checkpoint_file, to_use_gpu)
    print("...testing complete!")
    
    print("Writing results to file...")
    write_output(test_results, output_file)
    print("...output results written!")