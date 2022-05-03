#if !defined HW_2_1_H
#define HW_2_1_H
#include <complex>
#include <cmath>
#include <armadillo>

void set_preterm_beta(std::vector<Contraction> &bases);
    
void basis_functions(Molecule &m);

double factorial(int n);

double double_factorial(int n);

double combination(int n, int k);

std::vector<double> product_center(Contraction &f1, Contraction &f2, int k, int l);

double s_kl(Contraction &f1, Contraction &f2, int k, int l);

double contracted_overlap(Contraction &f1, Contraction &f2, double s_kl);

std::vector<double> normalization_constant(Contraction &c);

arma::mat overlap_matrix(std::vector<Contraction> &basis_funcs);

double erfc_poly_exp(double x);

double integral_6d(Contraction &a, Contraction &b, int k, int kprime, int l, int lprime);

arma::mat gamma(Molecule &m);

double ptot(std::vector<Contraction> &bases, arma::mat density_mat, int atom_id);

arma::mat fock(Molecule &m, arma::mat p_alpha, arma::mat p_beta, arma::mat overlap, arma::mat g);

arma::mat generate_p(std::vector<Contraction> &bases, int num_electrons, arma::mat evecs);

arma::mat generate_core_h(Molecule &m, arma::mat overlap, arma::mat g);

double cndo_energy(Molecule &m, arma::mat f_a, arma::mat f_b, arma::mat p_a, arma::mat p_b, arma::mat ham);

#endif