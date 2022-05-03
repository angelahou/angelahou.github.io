import random
import os
import torch
from sklearn.model_selection import train_test_split
from torch.utils.tensorboard import SummaryWriter
import time
import numpy as np
import pandas as pd

from ani_model import ANI

from aev_calc import calc_aev

def optimizer_to(optim, device):
    for param in optim.state.values():
        # Not sure there are any global tensors in the state dict
        if isinstance(param, torch.Tensor):
            param.data = param.data.to(device)
            if param._grad is not None:
                param._grad.data = param._grad.data.to(device)
        elif isinstance(param, dict):
            for subparam in param.values():
                if isinstance(subparam, torch.Tensor):
                    subparam.data = subparam.data.to(device)
                    if subparam._grad is not None:
                        subparam._grad.data = subparam._grad.data.to(device)
                        

def write_output(results, output_name):
    """
    Writes the predicted losses to a file.

    Inputs:
    -------
    results : dict
        A dictionary containing the epoch and validation losses 
        of the testing set
    output_name : str
        The file name to write to.
    """
    df = pd.DataFrame.from_dict(results)
    df.to_csv(output_name, index = False)
    
    print("Losses file written.")                       
                        

def train(data, arch_dict, 
          checkpoint_file='checkpoint.pt', 
          max_epochs=100, 
          validation_size=0.2, 
          batch_size=1024, 
          use_gpu=False, 
          random_state=42):
    tensorboard = SummaryWriter(log_dir = '', flush_secs=1, max_queue=1)

    #Standardize energies to mean 0, std 1
    for mol in data:
        mol["energies"] = \
            (np.array(mol["energies"]) - np.mean(mol["energies"])) / \
            np.std(mol["energies"])
    
    #Initialize NN
    model = ANI(arch_dict)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    start_epoch = 0
    
    random_state = np.random.RandomState(seed=random_state)
    
    #Split training and validation by molecules
    train_data, val_data = \
        train_test_split(data, 
                         test_size=validation_size, 
                         random_state=random_state)
    
    #Store epoch losses and validation losses
    epoch_losses = []
    val_losses = []
    
    if os.path.isfile(checkpoint_file):
        checkpoint = torch.load(checkpoint_file)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch']
        random_state = checkpoint['random']
        epoch_losses = checkpoint['epoch_losses']
        val_losses = checkpoint['val_losses']
        
    if use_gpu:
        model = model.to("cuda:0")
        optimizer_to(optimizer, 'cuda:0')
        
    index_molecules = range(len(train_data))
    total_conformations = \
        np.sum([train_data[i]["coordinates"].shape[0] 
                for i in range(len(train_data))])
    mapping = {"H": 0, "C": 1, "N": 2, "O": 3}
    

    for n_epoch in range(start_epoch, max_epochs):
        counter = 0
        if os.path.isfile(checkpoint_file):
            checkpoint = torch.load(checkpoint_file)
            counter = checkpoint['batch']
        #print('counter =', counter, 'conformation / batch size =', 
        #      np.floor(total_conformations / batch_size))
        epoch_loss = 0
        print('theoretical num_batches : ', 
              np.floor(total_conformations / batch_size)) 
              # , ' , actual num_batches : 5')
        while counter < np.floor(total_conformations / batch_size): 
            #(sum of the number of conformations for all the molecules) 
            # / batch_size => one epoch
            current_mol_idx = random_state.choice(index_molecules)
            current_molecule = train_data[current_mol_idx]
            all_conformation_idx = list(
                range(len(current_molecule["coordinates"]))
            )
            random_state.shuffle(all_conformation_idx)
            current_batch_idx = all_conformation_idx[:batch_size]
            batch_conformation = \
                current_molecule["coordinates"][current_batch_idx] 
                #batch_size, N_a, 3
            batch_energies = current_molecule["energies"][current_batch_idx] 
                #batch_size
            pred_energies = []
            square_diff = torch.tensor(0)
            if use_gpu:
                square_diff = square_diff.to('cuda:0')
            optimizer.zero_grad()
            for i in range(len(batch_conformation)):
#               curr_time = time.time()
                conformation = batch_conformation[i] #N_a, 3
                elements = np.array(
                    [mapping[atom] for atom in current_molecule["species"]]
                )
                aevs = [torch.tensor(
                    calc_aev(elements, conformation, a), dtype=torch.float
                ) 
                    for a in range(len(conformation))] #N_a, 384
#               print('aev time:', time.time() - curr_time)
                if use_gpu:
                    for j in range(len(aevs)):
                        aevs[j] = aevs[j].to('cuda:0')
                total_energy = model.forward(aevs, current_molecule["species"])
                square_diff = torch.add(square_diff, 
                                        (total_energy - batch_energies[i]) ** 2)
            loss = torch.div(square_diff, batch_size)
            epoch_loss += loss.detach().cpu().item()
            loss.backward()
            optimizer.step()
            counter += 1
            print("Epoch %d/%d - Batch %d: Loss value - %.4f" \
                  % (n_epoch + 1, max_epochs, counter, loss.item()))
            
            torch.save({
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch' : n_epoch,
            'random' : random_state,
            'batch' : counter,
            'epoch_losses' : epoch_losses,
            'val_losses' : val_losses
            }, 
            checkpoint_file)
        epoch_loss = epoch_loss / counter # Take average
        epoch_losses.append(epoch_loss)
        
        # Validate
        total_val_conformations = np.sum([val_data[i]["coordinates"].shape[0] 
                                          for i in range(len(val_data))])
        val_square_diff = torch.tensor(0)
        with torch.no_grad():
            for mol in val_data[:1]:
                pred_energies = []
                print(f'validating with {batch_size} conformations')
                for conformation in mol["coordinates"][:batch_size]:
                    elements = np.array([mapping[atom] 
                                         for atom in mol["species"]])
                    aevs = [torch.tensor(calc_aev(elements, conformation, a), 
                                         dtype=torch.float) 
                            for a in range(len(conformation))] #N_a, 384
                    if use_gpu:
                        for i in range(len(aevs)):
                            aevs[i] = aevs[i].to('cuda:0')
                    total_energy = model.forward(aevs, mol["species"])
                    pred_energies.append(total_energy.detach().cpu().item())
                val_square_diff = \
                    torch.add(val_square_diff, 
                              np.sum(
                                  (np.array(pred_energies) - \
                                   mol["energies"][:batch_size]) ** 2
                                    )
                             )
        
        val_loss = torch.div(val_square_diff, total_val_conformations)
        val_losses.append(val_loss.detach().cpu().item())
        
        results_dict = {'epoch_loss': epoch_losses, 'val_loss': val_losses}
        write_output(results_dict, 'losses.csv')
        
        tensorboard.add_scalar("epoch_losses", epoch_loss, n_epoch + 1)
        tensorboard.add_scalar("val_losses", val_loss, n_epoch + 1)
        
        print("Epoch %d/%d - Loss: %.3f - Val loss: %.3f" \
              % (n_epoch + 1, max_epochs, epoch_loss, val_loss))

        counter = 0
        torch.save({
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch' : n_epoch + 1,
            'random' : random_state,
            'batch' : counter,
            'epoch_losses' : epoch_losses,
            'val_losses' : val_losses
            }, checkpoint_file)

    return epoch_losses, val_losses

def predict_energies(test_data, arch_dict, batch_size, 
                     checkpoint_file='', 
                     use_gpu=False):
    
    #Standardize energies to mean 0, std 1
    for mol in test_data:
        mol["energies"] = \
            (np.array(mol["energies"]) - np.mean(mol["energies"])) / \
            np.std(mol["energies"])
    
    #Initialize NN
    model = ANI(arch_dict)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    if os.path.isfile(checkpoint_file):
        checkpoint = torch.load(checkpoint_file)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        
    if use_gpu:
            model = model.to("cuda:0")
            
    mapping={"H": 0, "C": 1, "N": 2, "O": 3}
    
    test_losses = []
    mol_energies = {}
    with torch.no_grad():
        for i in range(len(test_data)):
            pred_energies = []
            test_square_diff = torch.tensor(0)
            print(f'testing with {batch_size * 2} conformations')
            for conformation in test_data[i]["coordinates"][:batch_size*2]:
                elements=np.array(
                    [mapping[atom] for atom in test_data[i]["species"]]
                )
                aevs = [torch.tensor(calc_aev(elements, conformation, a), 
                        dtype=torch.float) 
                        for a in range(len(conformation))] #N_a, 384
                if use_gpu:
                    for j in range(len(aevs)):
                        aevs[j] = aevs[j].to('cuda:0')
                total_energy = model.forward(aevs, test_data[i]["species"])
                pred_energies.append(total_energy.detach().cpu().item())
            test_square_diff = \
                torch.add(test_square_diff, 
                          np.sum((np.array(pred_energies) - \
                                  test_data[i]["energies"][:batch_size*2]) **2))
            
            test_loss = torch.div(test_square_diff, 
                                  test_data[i]["coordinates"].shape[0])

            test_losses.append(test_loss.detach().cpu().item())
            mol_energies[i] = pred_energies
        
    return {'Test Losses' : test_losses, 'Molecular Energies' : mol_energies}
    
    