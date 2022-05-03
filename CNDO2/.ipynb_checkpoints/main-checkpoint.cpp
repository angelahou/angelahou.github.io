#include <iostream>
#include <math.h>
#include <armadillo>
#include <chrono>
#include "readfile.h"
#include "cndo.h"

int main(int argc, char *argv[])
{
    Molecule m;
    string fname = argv[1];
    Readfile(m, fname);
    
    // std::vector<Contraction> bases;
    auto t1 = std::chrono::high_resolution_clock::now();
    basis_functions(m);
    auto t2 = std::chrono::high_resolution_clock::now();
    std::cout << "Initializing basis functions took " << std::chrono::duration_cast<std::chrono::milliseconds>(t2-t1).count() << " ms." << std::endl;
    
    t1 = std::chrono::high_resolution_clock::now();
//     find normalization coefficients for each basis function
    for (int i = 0; i < m.bases.size(); i++) {
        m.bases[i].normalizations = normalization_constant(m.bases[i]);
    }
    for (int i = 0; i < m.s_orbitals.size(); i++) {
        m.s_orbitals[i].normalizations = normalization_constant(m.s_orbitals[i]);
    }
    t2 = std::chrono::high_resolution_clock::now();
    std::cout << "Normalizing basis functions took " << std::chrono::duration_cast<std::chrono::milliseconds>(t2-t1).count() << " ms." << std::endl;

    double tol = 1e-6;
    
    arma::mat p_alpha_old = arma::zeros<arma::mat>(m.bases.size(), m.bases.size());
    arma::mat p_beta_old = arma::zeros<arma::mat>(m.bases.size(), m.bases.size());
    
    arma::mat overlap = overlap_matrix(m.bases);
    arma::mat g = gamma(m);
    
    t1 = std::chrono::high_resolution_clock::now();
    arma::mat f_alpha = fock(m, p_alpha_old, p_beta_old, overlap, g);
    t2 = std::chrono::high_resolution_clock::now();
    std::cout << "Constructing one fock matrix took " << std::chrono::duration_cast<std::chrono::milliseconds>(t2-t1).count() << " ms." << std::endl;
    
    arma::mat f_beta = fock(m, p_beta_old, p_alpha_old, overlap, g);
    
    arma::vec eigvals_alpha;
    arma::mat eigvecs_alpha;
    t1 = std::chrono::high_resolution_clock::now();
    arma::eig_sym(eigvals_alpha, eigvecs_alpha, f_alpha);
    t2 = std::chrono::high_resolution_clock::now();
    std::cout << "Diagonalizing took " << std::chrono::duration_cast<std::chrono::milliseconds>(t2-t1).count() << " ms." << std::endl;
    
    arma::vec eigvals_beta;
    arma::mat eigvecs_beta;
    arma::eig_sym(eigvals_beta, eigvecs_beta, f_beta);
    
    t1 = std::chrono::high_resolution_clock::now();
    // arma::mat p_alpha_new = generate_p(m.bases, m.p, eigvecs_alpha);
    arma::mat p_alpha_new = eigvecs_alpha.cols(0, m.p - 1) * eigvecs_alpha.cols(0, m.p - 1).t();
    t2 = std::chrono::high_resolution_clock::now();
    std::cout << "Generating new density matrix took " << std::chrono::duration_cast<std::chrono::milliseconds>(t2-t1).count() << " ms." << std::endl;
    
    // arma::mat p_beta_new = generate_p(m.bases, m.q, eigvecs_beta);
    arma::mat p_beta_new = eigvecs_beta.cols(0, m.q - 1) * eigvecs_beta.cols(0, m.q - 1).t();
    int count = 1;
    while (!approx_equal(p_alpha_old, p_alpha_new, "absdiff", tol) && !approx_equal(p_beta_old, p_beta_new, "absdiff", tol)) {
        count++;
        
        p_alpha_old = p_alpha_new;
        p_beta_old = p_beta_new;
        
        f_alpha = fock(m, p_alpha_old, p_beta_old, overlap, g);
        f_beta = fock(m, p_beta_old, p_alpha_old, overlap, g);

        arma::eig_sym(eigvals_alpha, eigvecs_alpha, f_alpha);
        arma::eig_sym(eigvals_beta, eigvecs_beta, f_beta);

        // p_alpha_new = generate_p(m.bases, m.p, eigvecs_alpha);
        // p_beta_new = generate_p(m.bases, m.q, eigvecs_beta);
        p_alpha_new = eigvecs_alpha.cols(0, m.p - 1) * eigvecs_alpha.cols(0, m.p - 1).t();
        p_beta_new = eigvecs_beta.cols(0, m.q - 1) * eigvecs_beta.cols(0, m.q - 1).t();
    }
    
    std::cout << "There were " << count << " iterations." <<std::endl;

    t1 = std::chrono::high_resolution_clock::now();
    arma::mat core = generate_core_h(m, overlap, g);
    t2 = std::chrono::high_resolution_clock::now();
    std::cout << "Generating core hamiltonian took " << std::chrono::duration_cast<std::chrono::milliseconds>(t2-t1).count() << " ms." << std::endl;
    
    t1 = std::chrono::high_resolution_clock::now();
    double total_energy = cndo_energy(m, f_alpha, f_beta, p_alpha_new, p_beta_new, core);
    t2 = std::chrono::high_resolution_clock::now();
    std::cout << "Calculating total energy took " << std::chrono::duration_cast<std::chrono::milliseconds>(t2-t1).count() << " ms." << std::endl;
    std::cout << "Total energy: " << total_energy << "eV" << std::endl;
    
}