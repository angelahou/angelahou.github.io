import torch
from torch import nn

class Gaussian(nn.Module):
    """
    The Gaussian activation function between layers in the NN.
    """
    def forward(self, x):
        # return the Gaussian activation function
        return torch.exp(- x * x)

class ANI(nn.Module):
    """
    The overarching neural net architecture. Each ANI module has 4 subnets, one for each atom type.
    """
    def __init__(self, architecture_dict):
        super(ANI, self).__init__()
        self.subnets = nn.ModuleDict({'C':ANI_sub(architecture_dict['C']), 
                                      'H':ANI_sub(architecture_dict['H']),
                                      'O':ANI_sub(architecture_dict['O']),
                                      'N':ANI_sub(architecture_dict['N'])})
        
    def forward(self, aev_inputs, atom_types):
        """
        Feed forward function in the NN.

        Inputs:
        -------
        aev_inputs : np.array
            Atomic environment vectors as inputs to NN
        atom_types : np.array
            An array containing "C", "H", "O", or "N" 
            as descriptors for a molecule's structure

        Output:
        -------
        total_energy : tensor(dtype = torch.float)
            The predicted total energy of the molecule
        """
        # set the total_energy variable as a tensor with dtype float
        total_energy = torch.tensor(0)
        for atom in ["H", "C", "O", "N"]:
            if atom in atom_types:
                # pull out the AEV's matching a specific atom type
                atom_specific_aev = \
                    torch.stack([aev_inputs[idx] 
                                 for idx in range(len(atom_types)) 
                                 if atom_types[idx] == atom])

                # calculate each atom's contribution to energy
                atomic_energies = self.subnets[atom](atom_specific_aev)

                # cumulatively add that atom's energy contribution 
                # to the total energy
                total_energy = torch.add(total_energy,
                                         torch.sum(atomic_energies))
        return total_energy
    
class ANI_sub(nn.Module):
    """
    A subnetwork to supplement the overarching ANI NN. 
    Each subnetwork is responsible for predicting the energy contribution 
    from a single atom type.
    """
    def __init__(self, architecture):
        """
        Input:
        ------
        architecture : NN.Module
            a NN architecture. 
            In literature, it is defined as a fully connected NN 
            with 384 input nodes and two hidden layers. 
        """
        super(ANI_sub, self).__init__()
            
        self.layers = architecture
        
        for m in self.layers:
            if type(m) == nn.Linear:
                d = m.weight.shape[1] #number of input nodes

                # initialize weights between -1/d and 1/d
                nn.init.uniform_(m.weight, a=-1/d, b=1/d) 

                # initialize biases to 0
                nn.init.zeros_(m.bias) 
    
    def forward(self, x):
        return self.layers(x)