#ifndef READFILE_H
#define READFILE_H

#include <iostream>
#include <vector>
#include <math.h>
#include <armadillo>

using namespace std;

struct Contraction {
    int atom_id;
    int atom_num;
    int valence;
    double coords[3];
    std::vector<double> exponents;
    std::vector<double> coeffs;
    std::vector<double> normalizations;
    int orbital_direction;
    int l_a;
    arma::mat moms;
    double preterm;
    double neg_beta;
    
    double x;
    double y;
    double z;
    
    void set_moms(double l_a) {
        if (l_a == 1) {
            moms = arma::eye(3, 3);
        }
        else {
            moms = arma::zeros<arma::mat>(3, 3);
        }
    }
    
    void set_l_a(int l) {
        l_a = l;
    }
    
    void set_exponents(std::vector<double> &e) {
        for (int i = 0; i < e.size(); i++) {
            exponents.push_back(e[i]);
        }
    }
    
    void set_coeffs(std::vector<double> &c) {
        for (int i = 0; i < c.size(); i++) {
            coeffs.push_back(c[i]);
        }
    }
};

struct Molecule {
    int molec_charge;
    
    // std::vector<Atom> atoms;
    std::vector<Contraction> s_orbitals; // basis functions with only s orbitals
    std::vector<Contraction> bases; // basis functions with all orbitals
    std::vector<double> ptot;
    int p;
    int q;
    
    int num_basis_functions() {
        int carbons = 0;
        int hydrogens = 0;
        for (int i = 0; i < s_orbitals.size(); i++) {
            if (s_orbitals[i].atom_num == 1) {
                hydrogens++;
            }
            else {
                carbons++;
            }
        }
        return 4 * carbons + hydrogens;
    }

    int num_electrons() {
        int carbons = 0;
        int hydrogens = 0;
        for (int i = 0; i < s_orbitals.size(); i++) {
            if (s_orbitals[i].atom_num == 1) {
                hydrogens++;
            }
            else {
                carbons++;
            }
        }
        int num_basis =  4 * carbons + hydrogens;
        if (num_basis % 2 != 0) {
            throw std::runtime_error("must have an integer number of electrons");
        }
        return num_basis / 2;
    }
};


void Readfile(Molecule &molec, std::string &fname);

void Readfile_H(std::vector<double> &coeffs, std::vector<double> &exps, std::string &fname);

void Readfile_Shell2(std::vector<double> &coeffs_s, std::vector<double> &coeffs_p,
                std::vector<double> &exps, std::string &fname);

#endif // READFILE_H