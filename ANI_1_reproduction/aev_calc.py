import numpy as np


def calc_f_C(Rij, RC):
    """
    Distance conversion function. Returns 0 if RIJ > RC, 
    but returns a converted distance if RIJ >= RC

    Inputs:
    -------
    Rij : float
        distance between central atom i and all its nearest neighbors j
    RC : float
        the cutoff distance 
        (in literature, radial RC = 5.2A and angular RC = 3.5A)
    
    Output:
    -------
    f_C_value * indicator: float
        converted distance
    """
    # converted value assuming that Rij <= RC
    f_C_value = 0.5 * np.cos(np.pi * Rij / RC) + 0.5

    indicator = ((Rij <= RC) & (Rij != 0)).astype(float) 
    # Make f_C(0)=0 to make sure the sum in distance conversion function 
    # and radial conversion function can run with j=i

    return f_C_value * indicator

def radial_component(Rijs, eta, Rs, RC=5.2):
    """
    Calculates the radial environment component of an AEV.

    Inputs:
    -------
    Rijs : np.array
        an array of distances between central atom i and nearest neighbors j
    eta : float
        a free parameter (literature value is 16)
    Rs : np.array
        an array of distance parameters 
        generated from the original group's in-house NeuroChem software suite
    RC : float
        radial cutoff

    Output:
    -------
    the radial component for the AEV of a single conformation
    """

    # perform distance conversion first
    f_C_values = calc_f_C(Rijs, RC)

    # following the formula from literature, calculate within the summation
    individual_components = np.exp(-eta * (Rijs - Rs) ** 2) * f_C_values
    return np.sum(individual_components)

def angular_component(Rij_vectors, Rik_vectors, zeta, theta_s, eta, Rs, RC=3.5):
    """
    Calculates the angular environment component for a single AEV.
    
    Inputs:
    -------
    Rij_vectors : np.array
        2D array of distances 
        between central atom i and one of its nearest neighbors j
    Rik_vectors : np.array
        2D array of distances 
        between central atom i and one of its nearest neighbors k
    zeta : float
        free parameter (literature value is 32)
    theta_s : np.array
        an array of angular parameters
    eta : float
        free parameter (literature value is 16)
    Rs : np.array
        an array of distance parameters 
        generated from the original group's in-house NeuroChem software suite
    RC : float
        angular cutoff (literature value is 3.5)

    Output:
    -------
    the angular component for the AEV of a single conformation
    """
    
    # calculate theta_ijk values (angle between two vectors) 
    # from cosine, length of vectors, and their dot product
    dot_products = Rij_vectors.dot(Rik_vectors.T)
    Rij_norms = np.linalg.norm(Rij_vectors, axis=-1)
    Rik_norms = np.linalg.norm(Rik_vectors, axis=-1)
    norms = Rij_norms.reshape((-1, 1)).dot(Rik_norms.reshape((1, -1)))
    cos_values = np.clip(dot_products / (norms + 1e-8), -1, 1)
    theta_ijks = np.arccos(cos_values)
    theta_ijk_filter = (theta_ijks != 0).astype(float)

    # calculate mean distance between Rij and Rik
    mean_dists = (Rij_norms.reshape((-1, 1)) + Rik_norms.reshape((1, -1))) / 2

    # calculate distance conversions for both Rij and Rik vectors
    f_C_values_Rij = calc_f_C(Rij_norms, RC)
    f_C_values_Rik = calc_f_C(Rik_norms, RC)
    f_C_values = \
        f_C_values_Rij.reshape((-1, 1)).dot(f_C_values_Rik.reshape((1, -1)))

    # following the formula in literature, 
    # calculate everything within the summation
    individual_components = \
        (1 + np.cos(theta_ijks - theta_s)) ** zeta * \
        np.exp(-eta * (mean_dists - Rs) ** 2) * f_C_values * theta_ijk_filter
    return 2 ** (1 - zeta) * np.sum(individual_components)

def calc_aev(atom_types, coords, i_index):
    """
    Calculates the AEV.

    Inputs:
    -------
    atom_types : np.array
        1D array of the atom types, listed as integers. 
        For this study, we mapped the atoms C, H, O, and N into 0, 1, 2, 3.
    coords : np.array
        2D array of the coordinates of the molecule in question 
        (in one specific conformation)
    i_index : int
        the atom index that we are calculating the aev for
    """
    # first calculate the relative coordinates 
    # with respect to the central atom that we calculate aev for
    relative_coordinates = coords - coords[i_index]
    nearby_atom_indicator = np.linalg.norm(relative_coordinates, axis=-1) < 5.3
    relative_coordinates = relative_coordinates[nearby_atom_indicator]
    atom_types = atom_types[nearby_atom_indicator]

    # calculate radial and angular components respectively
    radial_aev = np.array(
        [radial_component(
            np.linalg.norm(relative_coordinates[atom_types == atom], axis=-1),
            eta, 
            Rs) \
         for atom in [0, 1, 2, 3] \
         for eta in [16] \
         for Rs in [0.900000, 1.168750, 1.437500, 1.706250,\
                    1.975000, 2.243750, 2.512500, 2.781250,\
                    3.050000, 3.318750, 3.587500, 3.856250,\
                    4.125000, 4.393750, 4.662500, 4.931250]
        ]
    ) # literature values hardcoded
    angular_aev = np.array(
        [angular_component(
            relative_coordinates[atom_types == atom_j], 
            relative_coordinates[atom_types == atom_k],
            zeta, 
            theta_s, 
            eta, 
            Rs) \
         for atom_j in [0, 1, 2, 3] \
         for atom_k in range(atom_j, 4) \
         for zeta in [32] \
         for theta_s in [0.19634954, 0.58904862, 0.9817477, 1.3744468,\
                         1.76714590, 2.15984490, 2.5525440, 2.9452430]\
         for eta in [8] \
         for Rs in [0.900000, 1.550000, 2.200000, 2.850000]
        ]
    ) # literature values hardcoded

    # the result is the concatenation of both radial and angular components
    return np.concatenate([radial_aev, angular_aev])