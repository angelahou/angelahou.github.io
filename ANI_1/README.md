# ANI_1
An attempt to reproduce the results of [Smith et al 2017](https://www.nature.com/articles/sdata2017193).

For reference, parameters to generate AEVs were obtained from the supplemental information from [Smith et al 2019](https://www.nature.com/articles/s41467-019-10827-4).

## Running the script
To train the dataset, run ```run_training.py```. Consider adding flags:
* ```-d```: the directory where the dataset lives.
* ```-c```: the directory where the checkpoint file lives. This is to ensure that if access to the GPU is interrupted, we can pick up where we left off.
* ```-v```: the fraction of the dataset to set aside for testing. By default, this value is set to 0.2.
* ```-e```: the maximum number of epochs to train. By default, this number is 100.
* ```-b```: the batch size. By default, this value is 1024.
* ```--use-gpu```: a boolean flag for whether to use the GPU.

For example, we could input into the terminal:

```run_training.py -d data/ANI-1_release/readers/ani_gdb_s03.h5 -c checkpoints/chkpt.pt --use-gpu```

Or, we could input multiple data file directories, separated by a comma:

```run_training.py -d data/ANI-1_release/readers/ani_gdb_s03.h5,data/ANI-1_release/readers/ani_gdb_s04.h5 -c checkpoints/chkpt.pt --use-gpu```