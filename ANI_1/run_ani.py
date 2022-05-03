import argparse
import os
import sys
import pickle


from pyanitools import anidataloader
import h5py

from sklearn.model_selection import train_test_split

from torch import nn

import train_test as tt

def write_testing_output(results, test_data, output_name):
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
    print("Pickling energies to: " + energies_csv_file)
    
    with open(energies_csv_file, 'wb') as of:
        pickle.dump(results['Molecular Energies'], of)
        
    print("Energies pickled.")
    
    # open a file to pickle test_data
    test_data_file = output_name + "_test_data.pickle"
    print("Pickling test_data to: " + test_data_file)
    
    with open(test_data_file, 'wb') as of:
        pickle.dump(test_data, of)
        
    print("test_data pickled.")
    
if __name__ == "__main__":
    # Define an argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data', 
                        required=True, 
                        dest='data_files', 
                        help='Paths of all h5 files to be loaded in, ' \
                             'separated by commas. E.g. ' \
                             '-d ../ANI-1_release/ani_gdb_s03.h5,' \
                             '../ANI-1_release/ani_gdb_s04.h5,' \
                             '../ANI-1_release/ani_gdb_s05.h5', 
                        type=str)
    parser.add_argument('-c', '--checkpoint', 
                        required=False, 
                        dest='cp_file', 
                        help='File containing the checkpoint ' \
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
    parser.add_argument('-t', '--test_size', 
                        required=False, 
                        dest='test_size', 
                        help='Decimal percentage for size of testing set '\
                             'by number of molecules. '\
                             'Must be in the range [0.0, 1.0). '\
                             'E.g. -t 0.2 means 80 percent of the molecules '\
                             'are used for training, '\
                             'and 20 percent for testing', 
                        type=float, 
                        default=0.2)
    parser.add_argument('-e', '--epochs', 
                        required=False, 
                        dest='max_epochs', 
                        help='Maximum number of epochs', 
                        type=int, 
                        default=100)
    parser.add_argument('-b', '--batch-size', 
                        required=False, 
                        dest ='batch_size', 
                        help='Size of each batch', 
                        type=int, 
                        default=1024)
    parser.add_argument('-o', '--output', 
                        required=False, 
                        dest='o_file', 
                        help='Prefix of output files to be written.', 
                        type=str, 
                        default="output")
    parser.add_argument('-r', '--random_state', 
                        required=False, 
                        dest='random_state', 
                        help='Random state of train/test split.', 
                        type=int, 
                        default=42)
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
    # output file to write to
    output_file = args.o_file
    random_state = args.random_state

    # assert that a portion of the data will be left out as validation
    val_size = args.val_size
    if val_size < 0 or val_size >= 1:
        sys.exit(
            "Please enter a validation size value in the range [0.0, 1.0)."
        )
    
    test_size = args.test_size
    if test_size < 0 or test_size >= 1:
        sys.exit(
            "Please enter a test size value in the range [0.0, 1.0)."
        )
        
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
    
    print('total_num_molecules : ', len(data_list))
    # split training and validation by molecules
    train_data, test_data = \
        train_test_split(data_list, 
                         test_size=test_size, 
                         random_state=random_state)

    # training the data
    print("Loading/Saving model to: " + checkpoint_file)
    print("Training until: " + \
          str(epochs) + \
          " epochs have been completed with batch size: " + \
          str(batchsize))
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
        tt.train(train_data, arch_dict, checkpoint_file, 
                 max_epochs=epochs, 
                 validation_size=val_size, 
                 batch_size=batchsize, 
                 use_gpu=to_use_gpu, 
                 random_state=random_state)

    print("...training complete!")
    print("Beginning testing...")
    test_results = \
        tt.predict_energies(test_data, 
                            arch_dict, 
                            batchsize, 
                            checkpoint_file, 
                            to_use_gpu)
    print("...testing complete!")
    
    print("Writing results to file...")
    write_testing_output(test_results, test_data, output_file)
    print("...output results written!")
    





