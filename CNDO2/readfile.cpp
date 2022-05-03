#include "readfile.h"
#include <stdexcept>
#include <fstream>
#include <iostream>
#include <sstream> //to use istringstream

using namespace std;

void Readfile(Molecule &molec, string &fname)
{
/*
    Reads in a text file containing number of atoms in the system, atomic number of each atom, and atom positions.
    Converts txt file contents into a vector of Atoms.
*/
    ifstream in;
    in.open(fname, ios::in);
    
    // throw an error if the file does not exist
    if (!in) { 
        throw std::runtime_error("file does not exist");
    }

    string line;
    Contraction read_atom;
    
    // read the first line and store the proclaimed number of atoms
    getline(in, line);
    
    int n_atoms;
    int molec_charge;
    istringstream iss(line);
    iss >> n_atoms >> molec.molec_charge >> molec.p >> molec.q;
    
    while (getline(in, line)) {
        istringstream iss(line);
        
        // check to make sure atom position format is correct
        if (!(iss >> read_atom.atom_num >> read_atom.x >> read_atom.y >> read_atom.z)) {
            throw std::runtime_error("atom format is incorrect");
        }

        
        // input coordinate positions into the Atom's coordinate variables
        iss >> read_atom.atom_num >> read_atom.x >> read_atom.y >> read_atom.z;
        read_atom.coords[0] = read_atom.x;
        read_atom.coords[1] = read_atom.y;
        read_atom.coords[2] = read_atom.z;
        
        molec.s_orbitals.push_back(read_atom);
    }
    
    // check to make sure proclaimed number of atoms matches the number of atom positions specified
    if (molec.s_orbitals.size() != n_atoms) {
        throw std::runtime_error("number of proclaimed atoms not consistent with number of atom positions specified");
    }
    
    in.close();
}

void Readfile_H(std::vector<double> &coeffs, std::vector<double> &exps, std::string &fname)
{
/*
    Reads in a text file containing contraction coefficients and alpha values for 1s orbital of H.
*/
    ifstream in;
    in.open(fname, ios::in);
    
    // throw an error if the file does not exist
    if (!in) { 
        throw std::runtime_error("file does not exist");
    }

    string line;
    double coeff;
    double exp;
    
    while (getline(in, line)) {
        istringstream iss(line);
        
        // check to make sure atom position format is correct
        if (!(iss >> exp >> coeff)) {
            throw std::runtime_error("atom format is incorrect");
        }

        // input coordinate positions into the Atom's coordinate variables
        iss >> exp >> coeff;
        coeffs.push_back(coeff);
        exps.push_back(exp);
    }
    in.close();
}

void Readfile_Shell2(std::vector<double> &coeffs_s, std::vector<double> &coeffs_p, 
                std::vector<double> &exps, std::string &fname)
{
/*
    Reads in a text file containing contraction coefficients and alpha values for 1s orbital of H.
*/
    ifstream in;
    in.open(fname, ios::in);
    
    // throw an error if the file does not exist
    if (!in) { 
        throw std::runtime_error("file does not exist");
    }

    string line;
    double coeff_s;
    double coeff_p;
    double exp;
    
    while (getline(in, line)) {
        istringstream iss(line);
        
        // check to make sure atom position format is correct
        if (!(iss >> exp >> coeff_s >> coeff_p)) {
            throw std::runtime_error("atom format is incorrect");
        }

        
        // input coordinate positions into the Atom's coordinate variables
        iss >> exp >> coeff_s >> coeff_p;
        coeffs_s.push_back(coeff_s);
        coeffs_p.push_back(coeff_p);
        exps.push_back(exp);
    }
    in.close();
}