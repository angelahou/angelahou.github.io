 // CPP program to illustrate
// concept of Virtual Functions

#include <iostream>
#include <math.h>
#include <armadillo>
#include <iterator>
#include "readfile.h"
#include "cndo.h"

using namespace std;

void set_preterm_beta(std::vector<Contraction> &c) {
    for (int i = 0; i < c.size(); i++) { // hydrogen
        if (c[i].atom_num == 1) {
            c[i].preterm = 7.176;
            c[i].neg_beta = 9;
        }
        else if (c[i].atom_num == 6) { // carbon
            if (c[i].l_a == 0) {
                c[i].preterm = 14.051;
            } else {
                c[i].preterm = 5.572;
            }
            c[i].neg_beta = 21;
        }
        else if (c[i].atom_num == 7) { // nitrogen
            if (c[i].l_a == 0) {
                c[i].preterm = 19.316;
            } else {
                c[i].preterm = 7.275;
            }
            c[i].neg_beta = 25;
        }
        else if (c[i].atom_num == 8) { // oxygen
            if (c[i].l_a == 0) {
                c[i].preterm = 25.390;
            } else {
                c[i].preterm = 9.111;
            }
            c[i].neg_beta = 31;
        } else { // atom num = 9, fluorine
            if (c[i].l_a == 0) {
                c[i].preterm = 32.272;
            } else {
                c[i].preterm = 11.080;
            }
            c[i].neg_beta = 39;
        }
    }
}

std::vector<double> set_ptot(Molecule &m, arma::mat density_mat) {
/*
    evaluates the total density matrix P^tot_AA for each atom of the molecule.
*/
    std::vector<double> result;
    double ptot_curr = 0;
    double atom_curr = 0;
    double basis_curr = 0;
    
    while ((atom_curr < m.s_orbitals.size()) && (basis_curr < m.bases.size())) {
        if (atom_curr == m.bases[basis_curr].atom_id) {
            ptot_curr += density_mat(basis_curr, basis_curr);
        }
        else {
            result.push_back(ptot_curr);
            ptot_curr = 0;
            atom_curr++;
            basis_curr--;
        }
        basis_curr++;
    }
    
    result.push_back(ptot_curr);
    
    return result;
}

void basis_functions(Molecule &m) {
    std::vector<Contraction> a = m.s_orbitals;
    
    std::vector<double> exps_h;
    std::vector<double> coeffs_h;
    std::string fname_h ("H_STO3G.txt");
    Readfile_H(coeffs_h, exps_h, fname_h);
    
    std::vector<double> exps_c;
    std::vector<double> coeffs_c_s;
    std::vector<double> coeffs_c_p;
    std::string fname_c;
    
    int vector_size = a.size();
    
    for (int i = 0; i < vector_size; i++) {
        if (a[i].atom_num == 1) {            
            Contraction addition;
            addition.valence = 1;
            addition.atom_id = i;
            addition.atom_num = a[i].atom_num;
            addition.set_exponents(exps_h);
            addition.set_coeffs(coeffs_h);
            addition.orbital_direction = 0;
            addition.set_l_a(0);
            addition.set_moms(0);
            // addition.coords[0] = a[i].x;
            // addition.coords[1] = a[i].y;
            // addition.coords[2] = a[i].z;
            std::copy(std::begin(a[i].coords), std::end(a[i].coords), std::begin(addition.coords));
            m.bases.push_back(addition);
            
            m.s_orbitals.push_back(addition);
        }
        else {
            if (a[i].atom_num == 6) {
                fname_c = "C_STO3G.txt";
            } else if (a[i].atom_num == 7) {
                fname_c = "N_STO3G.txt";
            } else if (a[i].atom_num == 8) {
                fname_c = "O_STO3G.txt";
            } else {
                fname_c = "F_STO3G.txt";
            }
            Readfile_Shell2(coeffs_c_s, coeffs_c_p, exps_c, fname_c);
            
            Contraction addition;
            addition.valence = a[i].atom_num - 2;
            addition.atom_num = a[i].atom_num;
            addition.atom_id = i;
            addition.set_exponents(exps_c);
            addition.set_coeffs(coeffs_c_s);
            addition.orbital_direction = 0;
            addition.set_l_a(0);
            addition.set_moms(0);
            // addition.coords[0] = a[i].x;
            // addition.coords[1] = a[i].y;
            // addition.coords[2] = a[i].z;
            std::copy(std::begin(a[i].coords), std::end(a[i].coords), std::begin(addition.coords));
            m.bases.push_back(addition);
            
            m.s_orbitals.push_back(addition);
            
            for (int j = 0; j < 3; j++) {
                Contraction addition;
                addition.atom_num = a[i].atom_num;
                addition.valence = addition.atom_num - 2;
                addition.atom_id = i;
                addition.exponents = exps_c;
                addition.coeffs = coeffs_c_p;
                // addition.coords[0] = a[i].x;
                // addition.coords[1] = a[i].y;
                // addition.coords[2] = a[i].z;
                std::copy(std::begin(a[i].coords), std::end(a[i].coords), std::begin(addition.coords));
                addition.orbital_direction = j;
                addition.set_l_a(1);
                addition.set_moms(1);
                m.bases.push_back(addition);
            }
        }
    }
    
    for (int i = 0; i < vector_size; i++) {
        m.s_orbitals.erase(m.s_orbitals.begin());
    }
    
    set_preterm_beta(m.bases);
    set_preterm_beta(m.s_orbitals);
}


double factorial(int n) {
    /*
        Calculates the factorial of given double n (n! = n * (n-1) ... * 1)
    */
    double product = 1.;
    while (n >=1) {
        product *= n;
        n--;
    }
    return product;
}

double double_factorial(int n) {
    /*
        Calculates the double factorial of given double n (n!! = n * (n-2) ... * 1)
    */
    double product = 1.;
    while (n >= 1) {
        product *= n;
        n -= 2;
    }
    return product;
}

double combination(int n, int k) {
    /*
        Calculates the combination of n choose k.
    */
    return factorial(n) / (factorial(k) * factorial(n-k));
}

std::vector<double> product_center(Contraction &f1, Contraction &f2, int k, int l) {
    /*
        Calculates the center of the product gaussian.
    */
    std::vector<double> product;
    for (int i = 0; i < 3; i++) {
        double value = (f1.exponents[k] * f1.coords[i] + f2.exponents[l] * f2.coords[i]) / (f1.exponents[k] + f2.exponents[l]);
        product.push_back(value);
    }
    return product;
}

double s_kl(Contraction &f1, Contraction &f2, int k, int l) {
    /*
        Calculates the primitive unnormalized overlap integral, S^(kl).
    */
    std::vector<double> center = product_center(f1, f2, k, l);
    
//     accumulate Sx, Sy, Sz. overlap = S^(AB)
    double overlap = 1.;
    for (int dim = 0; dim < 3; dim++) {
        double summation = 0.;
        double exp_preterm = exp(-f1.exponents[k] * f2.exponents[l] * pow(f1.coords[dim] - f2.coords[dim], 2) / (f1.exponents[k] + f2.exponents[l]));
        double sqrt_term = sqrt(M_PI / (f1.exponents[k] + f2.exponents[l]));

        for (int i = 0; i <= f1.moms(f1.orbital_direction, dim); i++) {
            for (int j = 0; j <= f2.moms(f2.orbital_direction, dim); j++) {
                if ((i + j)%2 == 0) {
                    double comb = combination(f1.moms(f1.orbital_direction, dim), i) * combination(f2.moms(f2.orbital_direction, dim), j);
                    double d_f = double_factorial(i + j - 1);
                    double pos_a = pow(center[dim] - f1.coords[dim], f1.moms(f1.orbital_direction, dim) - i);
                    double pos_b =  pow(center[dim] - f2.coords[dim], f2.moms(f2.orbital_direction, dim) - j);
                    double denominator = pow(2 * (f1.exponents[k] + f2.exponents[l]), double(i + j)/2.);
                    summation += comb * d_f * pos_a * pos_b / denominator;
                }
            }
        }

        summation = summation * exp_preterm * sqrt_term;
        overlap *= summation;
    }
    return overlap;
}

std::vector<double> normalization_constant(Contraction &c) {
    /*
        Calculates the overlap between a primitive gaussian with itself, to find normalization coefficients.
        orbital: 0 = x; 1 = y, 2 = z
    */
    std::vector<double> coefficients;
    for (int k = 0; k < 3; k++) {
        double overlap = s_kl(c, c, k, k);
        coefficients.push_back(1. / sqrt(overlap));
    }
    return coefficients;
}

double contracted_overlap(Contraction &f1, Contraction &f2) {
    double result = 0;
    for (int k = 0; k < 3; k++) {
        for (int l = 0; l < 3; l++) {
            double skl = s_kl(f1, f2, k, l);
            double term = f1.coeffs[k] * f2.coeffs[l] * f1.normalizations[k] * f2.normalizations[l] * skl;
            result += term;
        }
    }
    return result;
}

arma::mat overlap_matrix(std::vector<Contraction> &basis_funcs) {
    /*
        Calculates the matrix overlap between 2 primitive gaussian functions.
    */
    
    arma::mat overlap = arma::zeros<arma::mat>(basis_funcs.size(), basis_funcs.size());
    for (int i = 0; i < overlap.n_rows; i++) {
        for (int j = 0; j < overlap.n_cols; j++) {
            overlap(i, j) = contracted_overlap(basis_funcs[i], basis_funcs[j]);
        }
    }
    
    return overlap;
}

double erfc_poly_exp(double x)
{
    const int ncof = 5;
    const double cof[ncof] = {0.254829592, -0.284496736,  1.421413741,  -1.453152027, 1.061405429};
    double t = 1. / (1. + 0.3275911*x);
    double temp=t, sum=0.;
    for (int k= 0; k < ncof; k++){
        sum += cof[k] *temp;
        temp *= t;
    }
    return sum * std::exp(-x*x);
}

double integral_6d(Contraction &a, Contraction &b, int k, int kprime, int l, int lprime) {
/*
    calculates [0]^(0)
*/
    double sigma_a = pow(a.exponents[k] + a.exponents[kprime], -1.);
    double sigma_b = pow(b.exponents[l] + b.exponents[lprime], -1.);
    
    double ua = pow(M_PI * sigma_a, 1.5);
    double ub = pow(M_PI * sigma_b, 1.5);
    double numerator = ua * ub;
    
    // norm calculation
    double norm = 0;
    for (int dim = 0; dim < 3; dim++) {
        norm += pow(a.coords[dim] - b.coords[dim], 2.);
    }
    double denominator = sqrt(norm);    

    double v_squared = 1. / (sigma_a + sigma_b);
    double T = v_squared * norm;
    if (T == 0) {
        return numerator * sqrt(2. * v_squared) * sqrt(2. / M_PI);
    } else {
        // return numerator / denominator * erfc_poly_inv(sqrt(T));
        return numerator / denominator * (1 - erfc_poly_exp(sqrt(T)));
    }
}

arma::mat gamma(Molecule &m) {
/*
    returns the gamma matrix. Elements are converted into eV from au.
*/
    int n_atoms = m.s_orbitals.size();
    arma::mat values = arma::zeros<arma::mat>(n_atoms, n_atoms);
    
    for (int a = 0; a < values.n_rows; a++) {
        for (int b = 0; b < values.n_cols; b++) {
            double result = 0;
            for (int k = 0; k < 3; k++) {
                for (int kprime = 0; kprime < 3; kprime++) {
                    for (int l = 0; l < 3; l++) {
                        for (int lprime = 0; lprime < 3; lprime++) {
                            double k_coeff = m.s_orbitals[a].coeffs[k] * m.s_orbitals[a].normalizations[k];
                            double kprime_coeff = m.s_orbitals[a].coeffs[kprime] * m.s_orbitals[a].normalizations[kprime];
                            double l_coeff = m.s_orbitals[b].coeffs[l] * m.s_orbitals[b].normalizations[l];
                            double lprime_coeff = m.s_orbitals[b].coeffs[lprime] * m.s_orbitals[b].normalizations[lprime];
                            result += k_coeff * kprime_coeff * l_coeff * lprime_coeff * integral_6d(m.s_orbitals[a], m.s_orbitals[b], k, kprime, l, lprime);
                        }
                    }
                }
            }
            values(a, b) = result * 27.2113961;
        }
    }
    return values;
}

// double ptot(std::vector<Contraction> &bases, arma::mat density_mat, int atom_id) {
// /*
//     evaluates the total density matrix P^tot_AA, which is the sum of diag elements in density matrix for a single atom
// */
//     double result = 0;
//     for (int i = 0; i < density_mat.n_rows; i++) {
//         if (bases[i].atom_id == atom_id) {
//             result += density_mat(i, i);
//         }
//     }
//     return result;
// }

arma::mat fock(Molecule &m, arma::mat p_alpha, arma::mat p_beta, arma::mat overlap, arma::mat g) {
/*
    builds fock matrix, first with diagonal elements and then with off-diagonals.
*/
    
    arma::mat fock = arma::zeros<arma::mat>(m.bases.size(), m.bases.size());
    
    // diagonal elements
    arma::mat density_mat = p_alpha + p_beta;
    std::vector<double> ptot = set_ptot(m, density_mat);
    
    // need to incorporate atom id in there somehwere, its trying to access elems outside of range
    for (int i = 0; i < fock.n_rows; i++) {
        // double middle_term = ((ptot(m.bases, density_mat, m.bases[i].atom_id) - m.bases[i].valence) - (p_alpha(i, i) - 0.5))
        //                    * g(m.bases[i].atom_id, m.bases[i].atom_id);
        double middle_term = ((ptot[m.bases[i].atom_id] - m.bases[i].valence) - (p_alpha(i, i) - 0.5))
                           * g(m.bases[i].atom_id, m.bases[i].atom_id);
        double third_term = 0;
        for (int j = 0; j < m.s_orbitals.size(); j++) {
            if (m.bases[i].atom_id != m.s_orbitals[j].atom_id) {
                // third_term += (ptot(m.bases, density_mat, m.s_orbitals[j].atom_id) 
                //             - m.s_orbitals[j].valence) * g(m.bases[i].atom_id, m.bases[j].atom_id);
                third_term += (ptot[m.s_orbitals[j].atom_id]
                            - m.s_orbitals[j].valence) * g(m.bases[i].atom_id, m.bases[j].atom_id);
            }
        }
        fock(i, i) = -m.bases[i].preterm + middle_term + third_term;
    }
    
    // non diagonal elements
    for (int i = 0; i < fock.n_rows; i++) {
        for (int j = 0; j < fock.n_cols; j++) {
            if (i != j) {
                double first_off_diag = -0.5 * (m.bases[i].neg_beta + m.bases[j].neg_beta) * overlap(i, j);
                double second_off_diag = p_alpha(i, j) * g(m.bases[i].atom_id, m.bases[j].atom_id);
                // double first_off_diag = 0;
                fock(i, j) = first_off_diag - second_off_diag;
            }
        }
    }
    
    return fock;
}


// arma::mat generate_p(std::vector<Contraction> &bases, int num_electrons, arma::mat evecs) {
// /*
//     builds density matrix P given eigenvectors and the number of electrons to fill.
// */
//     arma::mat result = arma::zeros<arma::mat>(bases.size(), bases.size());
//     // for (int i = 0; i < result.n_rows; i++) {
//     //     for (int j = 0; j < result.n_cols; j++) {
//     //         double summation = 0;
//     //         for (int e = 0; e < num_electrons; e++) {
//     //             summation += evecs(i, e) * evecs(j, e);
//     //         }
//     //         result(i, j) = summation;
//     //     }
//     // }
//     for (int e = 0; e < num_electrons; e++) {
//         for (int i = 0; i < result.n_rows; i++) {
//             for (int j = 0; j < result.n_cols; j++) {
//                 result(i,j) += evecs(i, e) * evecs(j, e);
//             }
//         }
//     }
//     return result;
// }

arma::mat generate_core_h(Molecule &m, arma::mat overlap, arma::mat g) {
/*
    builds the core hamiltonian matrix.
*/
    arma::mat core = arma::zeros<arma::mat>(m.bases.size(), m.bases.size());
    
    for (int i = 0; i < core.n_rows; i++) {
        double middle = (m.bases[i].valence - 0.5) * g(m.bases[i].atom_id, m.bases[i].atom_id);
        double last = 0;
        double current_b = -1;
        for (int j = 0; j < core.n_cols; j++) {
            if ((m.bases[i].atom_id != m.bases[j].atom_id) && (m.bases[j].atom_id != current_b)) {
                last += m.bases[j].valence * g(m.bases[i].atom_id, m.bases[j].atom_id);
                current_b = m.bases[j].atom_id;
            }
        }
        core(i, i) = -m.bases[i].preterm - middle - last; 
    }
    
    for (int i = 0; i < core.n_rows; i++) {
        for (int j = 0; j < core.n_cols; j++) {
            if (i != j) {
                core(i, j) = -0.5 * (m.bases[i].neg_beta + m.bases[j].neg_beta) * overlap(i, j);
            }
        }
    }
    
    return core;
}

double cndo_energy(Molecule &m, arma::mat f_a, arma::mat f_b, arma::mat p_a, arma::mat p_b, arma::mat ham) {
/*
    calculates the molecule's energy with CNDO/2, given the final iteration of fock, hamiltonian, and density matrices.
*/
    double first = 0;
    double second = 0;
    for (int i = 0; i < m.bases.size(); i++) {
        for (int j = 0; j < m.bases.size(); j++) {
            first += p_a(i, j) * (ham(i, j) + f_a(i, j));
            second += p_b(i, j) * (ham(i, j) + f_b(i, j));
        }
    }
    
    std::cout << "Electron energy: " << 0.5 * (first + second) << "eV" << std::endl;
    
    double third = 0;
    double rab = 0;
    for (int i = 0; i < m.s_orbitals.size(); i++) {
        for (int j = 0; j < i; j++) {
            rab = sqrt(pow(m.s_orbitals[i].coords[0] - m.s_orbitals[j].coords[0], 2) 
                 + pow(m.s_orbitals[i].coords[1] - m.s_orbitals[j].coords[1], 2) 
                 + pow(m.s_orbitals[i].coords[2] - m.s_orbitals[j].coords[2], 2));
            third += m.s_orbitals[i].valence * m.s_orbitals[j].valence * 27.2113961 / rab;
        }
    }
    
    std::cout << "Nuclear repulsion energy: " << third << "eV" <<  std::endl;
    
    return 0.5 * (first + second) + third;
}