import os
import re
import numpy as np

def parse_loss_file(filename):
    f = open(filename, "r")
    line_count = 0
    for line in f.readlines():
        if line_count <= 10:
            if line.startswith("Training until:"):
                num_epochs = int(re.search(r"(\d+)", line).group())
            elif line.startswith("total_num_molecules"):
                num_mols = int(re.search(r"(\d+)", line).group())
            elif line.startswith("theoretical num_batches"):
                num_batches = int(re.search(r"(\d+)", line).group())
                losses = np.zeros((num_epochs, num_batches))
                epoch_losses = np.zeros((num_epochs, 2))
            line_count += 1
        else:
            if line.startswith("Epoch") and "Val" not in line:
                epoch, batch, loss = \
                    re.findall(r"Epoch (\d+)\/\d+ - .* (\d+): .* - ([.\d]+)", 
                               line)[0]
                losses[int(epoch) - 1, int(batch) - 1] = float(loss)
            elif "Val" in line:
                try:
                    one_epoch, epoch_tr_loss, epoch_va_loss = \
                        re.findall(r"Epoch (\d+)\/\d+ - .*: ([.\d]+) - .*: ([.\d]+)",
                                   line)[0]
                    epoch_losses[int(one_epoch) - 1] = np.array(
                        [float(epoch_tr_loss), float(epoch_va_loss)]
                    )
                except:
                    pass
    f.close()
    losses = np.ma.masked_equal(losses, 0)
    epoch_losses = np.ma.masked_equal(epoch_losses, 0)
    return {
        'num_mols': num_mols,
        'num_epochs': num_epochs,
        'num_batches': num_batches,
        'losses': losses,
        'epoch_losses': epoch_losses
    }