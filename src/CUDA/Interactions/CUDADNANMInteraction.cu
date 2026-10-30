//
// Created by jonah on 3/26/21.
//


#include "CUDADNANMInteraction.h"
#include "CUDA_DNANM.cuh"
#include "../../Interactions/DNANMInteraction.h"

#include <algorithm>
#include <fstream>
#include <set>
#include <sstream>
#include <vector>
#include <cstring>
#include <limits>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <cerrno>


CUDADNANMInteraction::CUDADNANMInteraction(bool btp) : DNANMInteraction(btp), CUDABaseInteraction() {
    
    _read_par = false;
    _angular = btp;
    //Not copied over to device memory
    _spring_potential = NULL;
    _spring_eqdist = NULL;
    _affected_len = NULL;

    //Copied over to device memory
    _h_ang_params = NULL;
    _d_ang_params = NULL;

    _h_ang_kbkt = NULL;
    _d_ang_kbkt = NULL;

    _h_affected_indx = NULL;
    _d_affected_indx = NULL;

    _h_affected = NULL;
    _d_affected = NULL;

    _h_aff_eqdist = NULL;
    _d_aff_eqdist = NULL;

    _h_aff_gamma = NULL;
    _d_aff_gamma = NULL;

    _spring_param_size_number = 0;
    _ang_param_size = 0;

    _edge_compatible = true;
    _use_debye_huckel = false;
    _use_oxDNA2_coaxial_stacking = false;
    _use_oxDNA2_FENE = false;
}


CUDADNANMInteraction::~CUDADNANMInteraction() {
    //Delete All pointers required for spring potential parameters
    if(_spring_potential != NULL) delete[] _spring_potential;
    if(_spring_eqdist != NULL) delete[] _spring_eqdist;

    if(_h_ang_params != NULL) delete[] _h_ang_params;
    if(_h_ang_kbkt != NULL) delete[] _h_ang_kbkt;
    if(_d_ang_params != NULL) CUDA_SAFE_CALL( cudaFree(_d_ang_params) );
    if(_d_ang_kbkt != NULL) CUDA_SAFE_CALL( cudaFree(_d_ang_kbkt) );

    if(_affected_len != NULL) delete[] _affected_len;

    if(_d_affected != NULL) CUDA_SAFE_CALL(cudaFree(_d_affected));
    if(_d_aff_gamma != NULL) CUDA_SAFE_CALL(cudaFree(_d_aff_gamma));
    if(_d_aff_eqdist != NULL) CUDA_SAFE_CALL(cudaFree(_d_aff_eqdist));
    if(_d_affected_indx != NULL) CUDA_SAFE_CALL( cudaFree(_d_affected_indx) );

    if(_h_affected != NULL) delete[] _h_affected;
    if(_h_aff_gamma != NULL) delete[] _h_aff_gamma;
    if(_h_aff_eqdist != NULL) delete[] _h_aff_eqdist;
    if(_h_affected_indx != NULL) delete[] _h_affected_indx;

    if(_d_anm_v9_physical_to_slot != nullptr) CUDA_SAFE_CALL(cudaFree(_d_anm_v9_physical_to_slot));
    if(_d_anm_v9_map_errors != nullptr) CUDA_SAFE_CALL(cudaFree(_d_anm_v9_map_errors));
    if(_d_is_strand_end != nullptr) {
        CUDA_SAFE_CALL(cudaFree(_d_is_strand_end));
    }
    if(_d_protein_charge != nullptr) {
        CUDA_SAFE_CALL(cudaFree(_d_protein_charge));
    }
    if(_d_protein_group != nullptr) {
        CUDA_SAFE_CALL(cudaFree(_d_protein_group));
    }
}


void CUDADNANMInteraction::get_settings(input_file &inp) {
    std::string inter_type;
    if (getInputString(&inp, "parfile", _parameterfile, 0) != KEY_FOUND) {
        throw oxDNAException("Key 'parfile' not found. Necessary for Protein sims.");
    }

    char s[5] = "none";
    if(strcmp(this->_parameterfile, s) != 0) _read_par = true;

    if(getInputString(&inp, "topology", this->_topology_filename, 0) != KEY_FOUND){
        throw oxDNAException("Key 'topology_file' not found.");
    }
    DNANMInteraction::get_settings(inp);
    // RC3 global DNACT is selected in the input, without environment opt-in.
    // Unsupported configurations cannot bypass this contract.
    if(_angular) {
        if(_parameter_kbkt || !_read_par)
            throw oxDNAException("DNACT CUDA requires GLOBAL stiffness and a validated ANM parameter file; per-bond stiffness is not qualified");
#ifdef CUDA_DOUBLE_PRECISION
        std::string dnact_precision = "double";
#else
        std::string dnact_precision = "mixed";
#endif
        getInputString(&inp, "backend_precision", dnact_precision, 0);
        if(dnact_precision != "mixed")
            throw oxDNAException("DNACT CUDA requires backend_precision = mixed in this release");
    }

    _use_debye_huckel = true;
    _use_oxDNA2_coaxial_stacking = true;
    _use_oxDNA2_FENE = true;

    // we don't need the F4_... terms as the macros are used in the CUDA_DNA.cuh file; this doesn't apply for the F2_K term
    F2_K[1] = CXST_K_OXDNA2;
    _debye_huckel_half_charged_ends = true;
    // C11: retain the grooving choice parsed by DNAInteraction::get_settings.
    // end copy from DNA2Interaction

    // copied from DNA2Interaction::get_settings() (CPU), the least bad way of doing things
    getInputNumber(&inp, "salt_concentration", &_salt_concentration, 1);
    getInputBool(&inp, "dh_half_charged_ends", &_debye_huckel_half_charged_ends, 0);

    // lambda-factor (the dh length at T = 300K, I = 1.0)
    _debye_huckel_lambdafactor = 0.3616455f;
    getInputFloat(&inp, "dh_lambda", &_debye_huckel_lambdafactor, 0);

    // the prefactor to the Debye-Huckel term
    _debye_huckel_prefactor = 0.0543f;
    getInputFloat(&inp, "dh_strength", &_debye_huckel_prefactor, 0);
    if(!std::isfinite(static_cast<double>(_salt_concentration)) ||
       _salt_concentration <= 0 ||
       !std::isfinite(static_cast<double>(_debye_huckel_lambdafactor)) ||
       _debye_huckel_lambdafactor <= 0 ||
       !std::isfinite(static_cast<double>(_debye_huckel_prefactor)) ||
       _debye_huckel_prefactor <= 0)
        throw oxDNAException("Invalid CUDA DNANM/DNACT Debye-Huckel input");

    if(this->_protein_dna_electrostatics || this->_protein_protein_electrostatics) {
        number requested_rhigh = (number) 0.;
        if(getInputNumber(&inp, "debye_huckel_rhigh", &requested_rhigh, 0) == KEY_FOUND)
            throw oxDNAException("CUDA protein-DNA electrostatics requires derived debye_huckel_rhigh");
    }


    // Only the unsorted execution path is qualified for this release.
    int anm_cuda_sort_interval = 0;
    getInputInt(&inp, "CUDA_sort_every", &anm_cuda_sort_interval, 0);
    if(anm_cuda_sort_interval != 0)
        throw oxDNAException("DNANM/DNACT CUDA requires CUDA_sort_every = 0 in this release");

}



void CUDADNANMInteraction::cuda_init(int N) {
    // Re-entering cuda_init() on a temperature update must not leak ANM CSR.
    if(_spring_potential != nullptr) { delete[] _spring_potential; _spring_potential = nullptr; }
    if(_spring_eqdist != nullptr) { delete[] _spring_eqdist; _spring_eqdist = nullptr; }
    if(_h_ang_params != nullptr) { delete[] _h_ang_params; _h_ang_params = nullptr; }
    if(_h_ang_kbkt != nullptr) { delete[] _h_ang_kbkt; _h_ang_kbkt = nullptr; }
    if(_affected_len != nullptr) { delete[] _affected_len; _affected_len = nullptr; }
    if(_h_affected != nullptr) { delete[] _h_affected; _h_affected = nullptr; }
    if(_h_aff_gamma != nullptr) { delete[] _h_aff_gamma; _h_aff_gamma = nullptr; }
    if(_h_aff_eqdist != nullptr) { delete[] _h_aff_eqdist; _h_aff_eqdist = nullptr; }
    if(_h_affected_indx != nullptr) { delete[] _h_affected_indx; _h_affected_indx = nullptr; }
    if(_d_ang_params != nullptr) { CUDA_SAFE_CALL(cudaFree(_d_ang_params)); _d_ang_params = nullptr; }
    if(_d_ang_kbkt != nullptr) { CUDA_SAFE_CALL(cudaFree(_d_ang_kbkt)); _d_ang_kbkt = nullptr; }
    if(_d_affected != nullptr) { CUDA_SAFE_CALL(cudaFree(_d_affected)); _d_affected = nullptr; }
    if(_d_aff_gamma != nullptr) { CUDA_SAFE_CALL(cudaFree(_d_aff_gamma)); _d_aff_gamma = nullptr; }
    if(_d_aff_eqdist != nullptr) { CUDA_SAFE_CALL(cudaFree(_d_aff_eqdist)); _d_aff_eqdist = nullptr; }
    if(_d_affected_indx != nullptr) { CUDA_SAFE_CALL(cudaFree(_d_affected_indx)); _d_affected_indx = nullptr; }

    CUDABaseInteraction::cuda_init(N);
    DNANMInteraction::init();

//    Addition of Reading Parameter File -> Moved from get_settings due to needing to fill variables that are filled in the CPU version of DNANMInteraction::read_topology
    std::fstream top;
    int tmp1, tmp2;
    top.open(this->_topology_filename, std::ios::in);
    if (top.is_open()){
        top >> tmp1 >> tmp2 >> this->ndna >> this->npro >> this->ndnas;
        top >> this->_firststrand;
        top.close();
    } else {
        throw oxDNAException("Could not open Topology File");
    }


    if(this->_firststrand < 0) offset = 0;
    else if(this->_firststrand > 0) offset = this->ndna;
    else throw oxDNAException("No Strand should have an ID of 0");

    // Fail closed on interleaved DNA/protein topology in *every* ES mode.
    if(this->ndna + this->npro != N || this->npro < 0 || this->ndna < 0)
        throw oxDNAException("ANM CUDA topology header particle counts inconsistent");
    {
        std::ifstream v9_topology(this->_topology_filename);
        if(!v9_topology) throw oxDNAException("Cannot open ANM CUDA topology for validation");
        std::string v9_row;
        if(!std::getline(v9_topology,v9_row))
            throw oxDNAException("Empty ANM CUDA topology");
        int v9_seen=0;
        while(std::getline(v9_topology,v9_row)) {
            std::istringstream v9_record(v9_row);
            int v9_strand=0;
            if(!(v9_record >> v9_strand)) continue;
            if(v9_seen >= N || v9_strand == 0)
                throw oxDNAException("ANM CUDA unexpected extra row or zero strand");
            const bool expected_protein = (v9_seen >= offset && v9_seen < offset+npro);
            if((v9_strand < 0) != expected_protein)
                throw oxDNAException("CUDA ANM expects all protein beads contiguous");
            ++v9_seen;
        }
        if(v9_seen != N) throw oxDNAException("ANM CUDA topology row-count mismatch");
    }
    if(_angular) {
        // CUDA edge lists omit n3/n5 pairs using the higher slot's list.
        // Require reciprocal unordered protein links so an omitted pair is
        // restored once at each endpoint. No CSR, adjacency or strand-number
        // constraint is needed. The completion kernel deduplicates n3/n5.
        std::ifstream angular_top(this->_topology_filename);
        std::string row;
        if(!angular_top || !std::getline(angular_top, row))
            throw oxDNAException("Cannot validate DNACT CUDA backbone topology");
        std::vector<int> n3(N, -1), n5(N, -1);
        int seen = 0;
        while(std::getline(angular_top, row)) {
            std::istringstream record(row);
            int strand;
            if(!(record >> strand)) continue;
            if(seen >= N) throw oxDNAException("Extra DNACT CUDA topology row");
            std::string residue;
            if(!(record >> residue >> n3[seen] >> n5[seen]))
                throw oxDNAException("Malformed DNACT CUDA backbone row");
            ++seen;
        }
        if(seen != N) throw oxDNAException("Incomplete DNACT CUDA backbone topology");
        for(int g = offset; g < offset + npro; ++g) {
            for(int target : {n3[g], n5[g]}) {
                if(target == -1) continue;
                if(target < offset || target >= offset + npro || target == g)
                    throw oxDNAException("Invalid DNACT CUDA backbone neighbour at bead %d", g);
                if(n3[target] != g && n5[target] != g)
                    throw oxDNAException("Nonreciprocal DNACT CUDA backbone neighbour at bead %d", g);
            }
        }
    }
    // Protein-DNA electrostatics: optional, with zero changes to the ES-OFF path.
    // The CUDA parent bypasses DNANMInteraction::read_topology(). Build its
    // residue metadata here and reuse the SAME strict CPU charge-file parser.
    if(_d_protein_charge != nullptr) {
        CUDA_SAFE_CALL(cudaFree(_d_protein_charge));
        _d_protein_charge = nullptr;
    }
    if(_d_protein_group != nullptr) {
        CUDA_SAFE_CALL(cudaFree(_d_protein_group));
        _d_protein_group = nullptr;
    }
    if(this->_protein_dna_electrostatics || this->_protein_protein_electrostatics) {
        this->_protein_local_to_global.clear();
        this->_protein_topology_strand.clear();
        this->_protein_residue_code.clear();
        std::ifstream electro_top(this->_topology_filename);
        if(!electro_top) throw oxDNAException("Cannot open topology for CUDA protein-DNA electrostatics");
        std::string line;
        if(!std::getline(electro_top, line)) throw oxDNAException("Empty DNANM topology");
        int seen = 0;
        while(std::getline(electro_top, line)) {
            std::istringstream record(line);
            int strand;
            if(!(record >> strand)) continue;
            if(strand == 0) throw oxDNAException("Unexpected strand zero in protein-DNA topology");
            std::string aa;
            if(!(record >> aa)) throw oxDNAException("Invalid residue in protein-DNA topology");
            if(strand < 0) {
                if(aa.size() != 1) throw oxDNAException("Electrostatic protein residue code must have one character");
                this->_protein_local_to_global.push_back(seen);
                this->_protein_topology_strand.push_back(strand);
                this->_protein_residue_code.push_back(static_cast<char>(
                    std::toupper(static_cast<unsigned char>(aa[0]))));
            }
            ++seen;
        }
        if(seen != N || static_cast<int>(this->_protein_local_to_global.size()) != this->npro)
            throw oxDNAException("CUDA electrostatic topology size mismatch");
        for(int li=0; li<this->npro; ++li)
            if(this->_protein_local_to_global[li] != offset+li)
                throw oxDNAException("CUDA ANM electrostatics requires contiguous protein topology block");
        this->_load_protein_charge_file(N);
        if(static_cast<int>(this->_protein_charge_reduced.size()) != N)
            throw oxDNAException("CUDA charge buffer length mismatch");
        const double factor = static_cast<double>(this->_protein_dna_dh_scale);
        if(!(factor >= 0.) || factor > static_cast<double>(std::numeric_limits<c_number>::max()))
            throw oxDNAException("Invalid CUDA protein-DNA electrostatic scaling");
        _cuda_protein_dna_dh_scale = static_cast<c_number>(factor);
        const double pp_factor = static_cast<double>(this->_protein_protein_dh_scale);
        if(!(pp_factor >= 0.) || !std::isfinite(pp_factor) ||
           pp_factor > static_cast<double>(std::numeric_limits<c_number>::max()))
            throw oxDNAException("Invalid CUDA protein-protein electrostatic scaling");
        _cuda_protein_protein_dh_scale = static_cast<c_number>(pp_factor);
        if(static_cast<int>(this->_protein_group.size()) != N)
            throw oxDNAException("CUDA protein group buffer length mismatch");
        std::vector<int> groups = this->_protein_group;
        std::vector<c_number> charges(N, (c_number) 0.f);
        for(int i=0;i<N;++i) {
            const double val = static_cast<double>(this->_protein_charge_reduced[i]);
            if(std::fabs(val) > static_cast<double>(std::numeric_limits<c_number>::max()))
                throw oxDNAException("CUDA reduced protein charge out of range at particle %d",i);
            charges[i] = static_cast<c_number>(val);
        }
        long double largest_charge = 0.0L;
        for(const c_number charge : charges) {
            largest_charge = std::max(largest_charge,
                std::fabs(static_cast<long double>(charge)));
        }
        const long double max_cuda =
            static_cast<long double>(std::numeric_limits<c_number>::max());
        if(this->_protein_dna_electrostatics &&
           static_cast<long double>(_cuda_protein_dna_dh_scale) * largest_charge > max_cuda)
            throw oxDNAException("CUDA protein-DNA charge product exceeds c_number range");
        if(this->_protein_protein_electrostatics &&
           static_cast<long double>(_cuda_protein_protein_dh_scale) * largest_charge * largest_charge > max_cuda)
            throw oxDNAException("CUDA protein-protein charge product exceeds c_number range");
        CUDA_SAFE_CALL(cudaMalloc(&_d_protein_charge, N*sizeof(c_number)));
        CUDA_SAFE_CALL(cudaMemcpy(_d_protein_charge, charges.data(),
                                 N*sizeof(c_number), cudaMemcpyHostToDevice));
        CUDA_SAFE_CALL(cudaMalloc(&_d_protein_group, N*sizeof(int)));
        CUDA_SAFE_CALL(cudaMemcpy(_d_protein_group, groups.data(),
                                 N*sizeof(int), cudaMemcpyHostToDevice));
    }



    if(_read_par){
        //Initalizing Some Host and Device Arrays for Spring Parameters
        _spring_param_size_number = sizeof(c_number) * (this->npro*this->npro);
        _ang_param_size = sizeof(c_number) * (this->npro*4);

        _spring_potential = new c_number[this->npro*this->npro]();
        _spring_eqdist = new c_number[this->npro*this->npro]();

        if(_angular) { // only need for anmt
            //Initializing Host and Device Arrays for Angular Parameters
            _h_ang_params = new c_number[this->npro*4]();
            CUDA_SAFE_CALL( cudaMalloc(&_d_ang_params, _ang_param_size));
            if (this->_parameter_kbkt) {
                _h_ang_kbkt = new c_number[this->npro * 4]();
                CUDA_SAFE_CALL(cudaMalloc(&_d_ang_kbkt, sizeof(c_number) * (this->npro * 2)));
            }
        }

        char potswitch = 'x';
        c_number potential = 0.f;
        c_number dist = 0.f;
        //initializing array members
        for(int i = 0; i< (this->npro*this->npro); i++){
            _spring_eqdist[i] = dist;
            _spring_potential[i] = potential;
            if(_angular) if(i < (this->npro *4 )) _h_ang_params[i] = dist;
        }

        //Checkers as Lambdas
        auto valid_angles = [](double a, double b, double c, double d)
        {
            double anglemin = std::min({a, b, c, d});
            double anglemax = std::max({a, b, c, d});
            if (anglemin < -1.0 || anglemax > 1.0){
                throw oxDNAException("Cos of Angle in Parameter File not in Valid bounds");
            }
        };

        auto valid_spring_params = [](int num_particles, int x, int y, double d, char s, double k){
            if(x < 0 || x >= num_particles) throw oxDNAException("Invalid Particle ID %d in Parameter File", x);
            if(y < 0 || y >= num_particles) throw oxDNAException("Invalid Particle ID %d in Parameter File", y);
            if(d < 0) throw oxDNAException("Invalid Eq Distance %d in Parameter File", d);
            if(s != 's') throw oxDNAException("Potential Type %c Not Supported", s);
            if(k < 0) throw oxDNAException("Spring Constant %f Not Supported", k);
        };



        //Reading Parameter File
        int key1, key2 = 0;
        c_number a0, b0, c0, d0;
        std::string carbons;

        //total connections
        int spring_connection_num = 0;
        std::set<std::pair<int,int>> anm_seen_edges;

        //allocate and declare affected_len vector
        _affected_len = new int[this->npro]();
        for(int i = 0; i < this->npro; i++) _affected_len[i] = 0;

        std::fstream parameters;
        parameters.open(this->_parameterfile, std::ios::in);
        getline (parameters,carbons);
        //Read Parameter File
        if (parameters.is_open())
        {
            while (parameters >> key1 >> key2 >> dist >> potswitch >> potential)
            {
                valid_spring_params(N, key1, key2, dist, potswitch, potential);
                spring_connection_num += 1;

                if(offset != 0) {
                    key1 -= offset;
                    key2 -= offset;
                }
                if(key1 < 0 || key1 >= this->npro ||
                   key2 < 0 || key2 >= this->npro)
                    throw oxDNAException("ANM spring endpoint outside protein-local range");
                if(!std::isfinite(static_cast<double>(dist)) || dist <= 0 ||
                   !std::isfinite(static_cast<double>(potential)) || potential < 0)
                    throw oxDNAException("Invalid ANM spring distance or stiffness");
                // Enforce one edge per unordered bead pair before updating
                // CSR counters; duplicates otherwise desynchronise offsets.
                if(key1 == key2)
                    throw oxDNAException("ANM self-spring is invalid");
                if(!anm_seen_edges.insert(std::minmax(key1,key2)).second)
                    throw oxDNAException("Duplicate ANM spring edge");

                _affected_len[key1] += 1;
                _affected_len[key2] += 1;
                //potswitch is currently unused but may be later

                if(_angular) {
                    if (key2 - key1 == 1) {
                        //Angular Parameters
                        parameters >> a0 >> b0 >> c0 >> d0;
                        valid_angles(a0, b0, c0, d0);
                        _h_ang_params[key1 * 4] = a0;
                        _h_ang_params[key1 * 4 + 1] = b0;
                        _h_ang_params[key1 * 4 + 2] = c0;
                        _h_ang_params[key1 * 4 + 3] = d0;


                        if (this->_parameter_kbkt) {
                            parameters >> _kbend >> _ktor;
                            if (_kbend < 0 || _ktor < 0)
                                throw oxDNAException("Invalid pairwise kb/kt Value Declared in Parameter File. Check Par Formatting");
                            _h_ang_kbkt[key1 * 2] = _kbend;
                            _h_ang_kbkt[key1 * 2 + 1] = _ktor;
                        }
                    }
                }

                _spring_potential[key1*this->npro + key2] = potential;
                _spring_eqdist[key1*this->npro + key2] = dist;

                _spring_potential[key2*this->npro + key1] = potential;
                _spring_eqdist[key2*this->npro + key1] = dist;

            }
            if(!parameters.eof())
                throw oxDNAException("Malformed or unsupported CUDA ANM parameter row; cannot silently truncate spring network");
            parameters.close();
        } else {
            throw oxDNAException("ParameterFile Could Not Be Opened");
        }

        if (spring_connection_num == 0) {
            throw oxDNAException("CUDA ANM zero-spring file is not qualified; use parfile=none");
        }
        //Compressed Parameter Initialization
        _h_affected_indx = new int[this->npro + 1]();
        _h_affected = new int[spring_connection_num*2]();
        _h_aff_gamma = new c_number[spring_connection_num*2]();
        _h_aff_eqdist = new c_number[spring_connection_num*2]();
        auto zero = (c_number) 0.f;
        for(int i = 0; i < this->npro+1; i++) _h_affected_indx[i] = 0;
        for(int i = 0; i < spring_connection_num*2; i++){
            _h_affected[i] = 0;
            _h_aff_gamma[i] = zero;
            _h_aff_eqdist[i] = zero;
        }

        //Compressed Index
        int param_indx = 0;
        //For each residue
        for(int i = 0; i < this->npro; i++){
            //Fill _h_affected filtering through larger arrays filled in parameter file reading
            for(int j = i*this->npro; j < i*this->npro+this->npro; j++){
                if(_spring_eqdist[j] != 0.f){
                    //Affected List, Access is controlled with indices in _h_affected_indx
                    _h_affected[param_indx] = j % this->npro;
                    //Stored in same way for easy access, spring constants
                    _h_aff_gamma[param_indx] = _spring_potential[j];
                    //eq_distance
                    _h_aff_eqdist[param_indx] = _spring_eqdist[j];
                    param_indx += 1;
                }
            }
        }

        //Don't need Larger arrays anymore, safe to delete
        if(_spring_eqdist != NULL) delete[] _spring_eqdist;
        _spring_eqdist = NULL; //Otherwise dangling Pointer
        if(_spring_potential != NULL) delete[] _spring_potential;
        _spring_potential = NULL;

        //Allocation and Copying of Compressed Parameters
        CUDA_SAFE_CALL(cudaMalloc(&_d_affected, 2 * spring_connection_num * sizeof(int)));
        CUDA_SAFE_CALL(cudaMemcpy(_d_affected, _h_affected, 2 * spring_connection_num * sizeof(int), cudaMemcpyHostToDevice));

        CUDA_SAFE_CALL(cudaMalloc(&_d_aff_gamma, 2 * spring_connection_num * sizeof(c_number)));
        CUDA_SAFE_CALL(cudaMemcpy(_d_aff_gamma, _h_aff_gamma, 2 * spring_connection_num * sizeof(c_number), cudaMemcpyHostToDevice));

        CUDA_SAFE_CALL(cudaMalloc(&_d_aff_eqdist, 2 * spring_connection_num * sizeof(c_number)));
        CUDA_SAFE_CALL(cudaMemcpy(_d_aff_eqdist, _h_aff_eqdist, 2 * spring_connection_num * sizeof(c_number), cudaMemcpyHostToDevice));

        int ind = 0;
        _h_affected_indx[0] = 0;
        //make indx access list where: _h_affected_indx[i] lower bound of i's parameters, _h_affected_indx[i+1] upper bound of i's parameters
        for(int i = 0; i < this->npro; i++){
            ind += _affected_len[i];
            _h_affected_indx[i+1] += ind;
        }

        //Don't need this anymore
        if(_affected_len != NULL) delete[] _affected_len;
        _affected_len = NULL;

        //Allocation and copying of Indice List for accessing compressed parameters
        CUDA_SAFE_CALL(cudaMalloc(&_d_affected_indx, (this->npro+1)*sizeof(int)));
        CUDA_SAFE_CALL(cudaMemcpy(_d_affected_indx, _h_affected_indx, (this->npro+1)*sizeof(int), cudaMemcpyHostToDevice));

        if(_angular) {
            //Parameters for Bending/Torsional, _h_ang_params is filled in parameter file reading
            CUDA_SAFE_CALL(cudaMemcpy(_d_ang_params, _h_ang_params, _ang_param_size, cudaMemcpyHostToDevice));
            if (this->_parameter_kbkt) {
                CUDA_SAFE_CALL(cudaMemcpy(_d_ang_kbkt, _h_ang_kbkt, sizeof(c_number) * (this->npro * 2),
                                          cudaMemcpyHostToDevice));
            }
            float param_memory_mb;
            //Memory Used by Parameters
            if (this->_parameter_kbkt) {
                param_memory_mb =
                        (spring_connection_num * 2 * sizeof(int) + 2 * spring_connection_num * 2 * sizeof(c_number)
                         + (this->npro + 1) * sizeof(int) + 6 * this->npro * sizeof(c_number)) / SQR(1024);
            } else {
                param_memory_mb =
                        (spring_connection_num * 2 * sizeof(int) + 2 * spring_connection_num * 2 * sizeof(c_number)
                         + (this->npro + 1) * sizeof(int) + 4 * this->npro * sizeof(c_number)) / SQR(1024);
            }
            OX_LOG(Logger::LOG_INFO, "Spring Parameters Size: %.2f MB", param_memory_mb);
        }

    } else OX_LOG(Logger::LOG_INFO, "Parfile: NONE, No protein parameters were filled");

    float f_copy = this->_hb_multiplier;
    CUDA_SAFE_CALL( cudaMemcpyToSymbol(MD_hb_multi, &f_copy, sizeof(float)) );
    CUDA_SAFE_CALL( cudaMemcpyToSymbol(MD_N, &N, sizeof(int)) );

    c_number tmp[50];
    for(int i = 0; i < 2; i++) for(int j = 0; j < 5; j++) for(int k = 0; k < 5; k++) tmp[i*25 + j*5 + k] = this->F1_EPS[i][j][k];

    COPY_ARRAY_TO_CONSTANT(MD_F1_EPS, tmp, 50);

    for(int i = 0; i < 2; i++) for(int j = 0; j < 5; j++) for(int k = 0; k < 5; k++) tmp[i*25 + j*5 + k] = this->F1_SHIFT[i][j][k];

    COPY_ARRAY_TO_CONSTANT(MD_F1_SHIFT, tmp, 50);

    COPY_ARRAY_TO_CONSTANT(MD_F1_A, this->F1_A, 2);
    COPY_ARRAY_TO_CONSTANT(MD_F1_RC, this->F1_RC, 2);
    COPY_ARRAY_TO_CONSTANT(MD_F1_R0, this->F1_R0, 2);
    COPY_ARRAY_TO_CONSTANT(MD_F1_BLOW, this->F1_BLOW, 2);
    COPY_ARRAY_TO_CONSTANT(MD_F1_BHIGH, this->F1_BHIGH, 2);
    COPY_ARRAY_TO_CONSTANT(MD_F1_RLOW, this->F1_RLOW, 2);
    COPY_ARRAY_TO_CONSTANT(MD_F1_RHIGH, this->F1_RHIGH, 2);
    COPY_ARRAY_TO_CONSTANT(MD_F1_RCLOW, this->F1_RCLOW, 2);
    COPY_ARRAY_TO_CONSTANT(MD_F1_RCHIGH, this->F1_RCHIGH, 2);

    COPY_ARRAY_TO_CONSTANT(MD_F2_K, this->F2_K, 2);
    COPY_ARRAY_TO_CONSTANT(MD_F2_RC, this->F2_RC, 2);
    COPY_ARRAY_TO_CONSTANT(MD_F2_R0, this->F2_R0, 2);
    COPY_ARRAY_TO_CONSTANT(MD_F2_BLOW, this->F2_BLOW, 2);
    COPY_ARRAY_TO_CONSTANT(MD_F2_BHIGH, this->F2_BHIGH, 2);
    COPY_ARRAY_TO_CONSTANT(MD_F2_RLOW, this->F2_RLOW, 2);
    COPY_ARRAY_TO_CONSTANT(MD_F2_RHIGH, this->F2_RHIGH, 2);
    COPY_ARRAY_TO_CONSTANT(MD_F2_RCLOW, this->F2_RCLOW, 2);
    COPY_ARRAY_TO_CONSTANT(MD_F2_RCHIGH, this->F2_RCHIGH, 2);

    COPY_ARRAY_TO_CONSTANT(MD_F5_PHI_A, this->F5_PHI_A, 4);
    COPY_ARRAY_TO_CONSTANT(MD_F5_PHI_B, this->F5_PHI_B, 4);
    COPY_ARRAY_TO_CONSTANT(MD_F5_PHI_XC, this->F5_PHI_XC, 4);
    COPY_ARRAY_TO_CONSTANT(MD_F5_PHI_XS, this->F5_PHI_XS, 4);


    if(this->_use_edge) CUDA_SAFE_CALL(cudaMemcpyToSymbol(MD_n_forces, &this->_n_forces, sizeof(int)));

    // copied from DNA2Interaction::init() (CPU), the least bad way of doing things
    // We wish to normalise with respect to T=300K, I=1M. 300K=0.1 s.u. so divide this->_T by 0.1
    c_number lambda = _debye_huckel_lambdafactor * sqrt(this->_T / 0.1f) / sqrt(_salt_concentration);
    // RHIGH gives the distance at which the smoothing begins
    _debye_huckel_RHIGH = 3.0 * lambda;
    _minus_kappa = -1.0 / lambda;

    // these are just for convenience for the smoothing parameter computation
    c_number x = _debye_huckel_RHIGH;
    c_number q = _debye_huckel_prefactor;
    c_number l = lambda;

    // compute the some smoothing parameters
    _debye_huckel_B = -(exp(-x / l) * q * q * (x + l) * (x + l)) / (-4. * x * x * x * l * l * q);
    _debye_huckel_RC = x * (q * x + 3. * q * l) / (q * (x + l));

    c_number debyecut;
    if (this->_grooving) {
        debyecut = 2.0f * sqrt(SQR(POS_MM_BACK1) + SQR(POS_MM_BACK2)) + _debye_huckel_RC;
    } else {
        debyecut = 2.0f * sqrt(SQR(POS_BACK)) + _debye_huckel_RC;
    }
    // the cutoff radius for the potential should be the larger of rcut and debyecut
    if (debyecut > this->_rcut) {
        this->_rcut = debyecut;
        this->_sqr_rcut = debyecut * debyecut;
    }
    // End copy from DNA2Interaction

    CUDA_SAFE_CALL(cudaMemcpyToSymbol(MD_dh_RC, &_debye_huckel_RC, sizeof(float)));
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(MD_dh_RHIGH, &_debye_huckel_RHIGH, sizeof(float)));
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(MD_dh_prefactor, &_debye_huckel_prefactor, sizeof(float)));
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(MD_dh_B, &_debye_huckel_B, sizeof(float)));
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(MD_dh_minus_kappa, &_minus_kappa, sizeof(float)));
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(MD_dh_half_charged_ends, &_debye_huckel_half_charged_ends, sizeof(bool)));

    //Constants for DNA/Protein Excluded Volume Interactions
    //Backbone-Protein Excluded Volume Parameters
    _pro_backbone_sigma = 0.57f;
    _pro_backbone_rstar= 0.569f;
    _pro_backbone_b = 178699253.5f;
    _pro_backbone_rcut = 0.572934f;
    _pro_backbone_sqr_rcut = 0.3283f;
    //Base-Protein Excluded Volume Parameters
    _pro_base_sigma = 0.36f;
    _pro_base_rstar= 0.359f;
    _pro_base_b = 296866090.0f;
    _pro_base_rcut = 0.362897f;
    _pro_base_sqr_rcut = 0.1317f;
    //Protein-Protein Excluded Volume Parameters
    _pro_sigma = 0.35f;
    _pro_rstar = 0.349f;
    _pro_b = 306484596.0f;
    _pro_rcut = 0.352894;
    _pro_sqr_rcut = 0.12454f; //_rc ^2

    if(_angular) {
        if (!this->_parameter_kbkt) {
            _kbend = this->_k_bend;
            _ktor = this->_k_tor;

            //kb and kt Parameters
            CUDA_SAFE_CALL(cudaMemcpyToSymbol(_kb, &_kbend, sizeof(float)));
            CUDA_SAFE_CALL(cudaMemcpyToSymbol(_kt, &_ktor, sizeof(float)));
        }
    }

    CUDA_SAFE_CALL( cudaMemcpyToSymbol(MD_pro_sigma, &_pro_sigma, sizeof(float)) );
    CUDA_SAFE_CALL( cudaMemcpyToSymbol(MD_pro_rstar, &_pro_rstar, sizeof(float)) );
    CUDA_SAFE_CALL( cudaMemcpyToSymbol(MD_pro_rc, &_pro_rcut, sizeof(float)) );
    CUDA_SAFE_CALL( cudaMemcpyToSymbol(MD_pro_b, &_pro_b, sizeof(float)) );
    CUDA_SAFE_CALL( cudaMemcpyToSymbol(MD_pro_sqr_rcut, &_pro_sqr_rcut, sizeof(float)) );

    CUDA_SAFE_CALL( cudaMemcpyToSymbol(MD_pro_backbone_sigma, &_pro_backbone_sigma, sizeof(float)) );
    CUDA_SAFE_CALL( cudaMemcpyToSymbol(MD_pro_backbone_rstar, &_pro_backbone_rstar, sizeof(float)) );
    CUDA_SAFE_CALL( cudaMemcpyToSymbol(MD_pro_backbone_rc, &_pro_backbone_rcut, sizeof(float)) );
    CUDA_SAFE_CALL( cudaMemcpyToSymbol(MD_pro_backbone_b, &_pro_backbone_b, sizeof(float)) );
    CUDA_SAFE_CALL( cudaMemcpyToSymbol(MD_pro_backbone_sqr_rcut, &_pro_backbone_sqr_rcut, sizeof(float)) );
    const float pd_backbone_stiffness_device = static_cast<float>(_pro_backbone_stiffness);
    CUDA_SAFE_CALL( cudaMemcpyToSymbol(MD_pro_backbone_stiffness, &pd_backbone_stiffness_device, sizeof(float)) );

    CUDA_SAFE_CALL( cudaMemcpyToSymbol(MD_pro_base_sigma, &_pro_base_sigma, sizeof(float)) );
    CUDA_SAFE_CALL( cudaMemcpyToSymbol(MD_pro_base_rstar, &_pro_base_rstar, sizeof(float)) );
    CUDA_SAFE_CALL( cudaMemcpyToSymbol(MD_pro_base_rc, &_pro_base_rcut, sizeof(float)) );
    CUDA_SAFE_CALL( cudaMemcpyToSymbol(MD_pro_base_b, &_pro_base_b, sizeof(float)) );
    CUDA_SAFE_CALL( cudaMemcpyToSymbol(MD_pro_base_sqr_rcut, &_pro_base_sqr_rcut, sizeof(float)) );
    const float pd_base_stiffness_device = static_cast<float>(_pro_base_stiffness);
    CUDA_SAFE_CALL( cudaMemcpyToSymbol(MD_pro_base_stiffness, &pd_base_stiffness_device, sizeof(float)) );

    CUDA_SAFE_CALL( cudaMemcpyToSymbol(MD_dna_sqr_rcut, &this->_sqr_rcut, sizeof(float)) );

    //Parameters for DNANM book keeping
    CUDA_SAFE_CALL( cudaMemcpyToSymbol(_ndna, &this->ndna, sizeof(int)) );
    CUDA_SAFE_CALL( cudaMemcpyToSymbol(_npro, &this->npro, sizeof(int)) );
    CUDA_SAFE_CALL( cudaMemcpyToSymbol(_offset, &this->offset, sizeof(int)) );
    // RC1: physics potential always accumulated; no runtime energy mask.

    if(!_anm_v9_list_cutoff_recorded) {
        _anm_v9_initial_list_cutoff = get_cuda_rcut();
        _anm_v9_list_cutoff_recorded = true;
    }
}

void CUDADNANMInteraction::compute_forces(CUDABaseList *lists, c_number4 *d_poss, GPU_quat *d_orientations, c_number4 *d_forces, c_number4 *d_torques, LR_bonds *d_bonds, CUDABox *d_box) {
    // Reconstruct and validate a bijective physical-ID -> CUDA-slot map
    // on every force call, including immediately after Hilbert sorting.
    if(_d_anm_v9_physical_to_slot == nullptr) {
        CUDA_SAFE_CALL(cudaMalloc(&_d_anm_v9_physical_to_slot, sizeof(int) * _N));
    }
    if(_d_anm_v9_map_errors == nullptr) {
        CUDA_SAFE_CALL(cudaMalloc(&_d_anm_v9_map_errors, 2 * sizeof(int)));
    }
    CUDA_SAFE_CALL(cudaMemset(_d_anm_v9_physical_to_slot, 0xff, sizeof(int) * _N));
    CUDA_SAFE_CALL(cudaMemset(_d_anm_v9_map_errors, 0, 2 * sizeof(int)));
    dnanm_v9_build_slotmap<<<_launch_cfg.blocks, _launch_cfg.threads_per_block>>>(
        d_poss, _d_anm_v9_physical_to_slot, _d_anm_v9_map_errors, _N);
    dnanm_v9_check_slotmap<<<_launch_cfg.blocks, _launch_cfg.threads_per_block>>>(
        d_poss, _d_anm_v9_physical_to_slot, _d_anm_v9_map_errors, _N);
    CUDA_SAFE_CALL(cudaGetLastError());
    int anm_v9_map_errors[2] = {0,0};
    CUDA_SAFE_CALL(cudaMemcpy(anm_v9_map_errors, _d_anm_v9_map_errors,
                              2 * sizeof(int), cudaMemcpyDeviceToHost));
    if(anm_v9_map_errors[0] != 0 || anm_v9_map_errors[1] != 0)
        throw oxDNAException("ANM sorting invalid physical-ID map: duplicates/invalid=%d, missing=%d",
                              anm_v9_map_errors[0], anm_v9_map_errors[1]);
    // DNA termini are slot-addressed by nonbonded kernels: update after sort.
    if(_d_is_strand_end == nullptr) _init_strand_ends(d_bonds);
    else {
        dna_init_DNA_strand_ends<<<_launch_cfg.blocks, _launch_cfg.threads_per_block>>>(
            _d_is_strand_end, d_bonds, _N);
        CUDA_SAFE_CALL(cudaGetLastError());
    }
    if(_d_is_strand_end == nullptr) {
        _init_strand_ends(d_bonds);
    }


    if(_update_st) {
        CUDA_SAFE_CALL(cudaMemset(_d_st, 0, _N * sizeof(CUDAStressTensor)));
    }

    if(_use_edge) {

        if(_n_forces == 1) { // we can directly use d_forces and d_torques so that no sum is required
            dnanm_forces_edge_nonbonded
            <<<(lists->N_edges - 1)/(_launch_cfg.threads_per_block) + 1, _launch_cfg.threads_per_block>>>
                    (d_poss, d_orientations, d_forces, d_torques, lists->d_edge_list, lists->N_edges, _d_is_strand_end,
                     _grooving, _use_debye_huckel, _use_oxDNA2_coaxial_stacking,
                     _d_protein_charge, _d_protein_group,
                     this->_protein_dna_electrostatics, this->_protein_protein_electrostatics,
                     this->_exclude_intragroup_electrostatics,
                     _cuda_protein_dna_dh_scale, _cuda_protein_protein_dh_scale,
                     _d_affected_indx, _d_affected,
                     _update_st, _d_st, d_box);
        }
        else { // sum required, somewhat slower
            dnanm_forces_edge_nonbonded
            <<<(lists->N_edges - 1)/(_launch_cfg.threads_per_block) + 1, _launch_cfg.threads_per_block>>>
                    (d_poss, d_orientations, _d_edge_forces, _d_edge_torques, lists->d_edge_list, lists->N_edges, _d_is_strand_end,
                     _grooving, _use_debye_huckel, _use_oxDNA2_coaxial_stacking,
                     _d_protein_charge, _d_protein_group,
                     this->_protein_dna_electrostatics, this->_protein_protein_electrostatics,
                     this->_exclude_intragroup_electrostatics,
                     _cuda_protein_dna_dh_scale, _cuda_protein_protein_dh_scale,
                     _d_affected_indx, _d_affected,
                     _update_st, _d_st, d_box);

            _sum_edge_forces_torques(d_forces, d_torques);
        }

        dna_forces_edge_bonded_dnanm
        <<<_launch_cfg.blocks, _launch_cfg.threads_per_block>>>
                (d_poss, d_orientations, d_forces, d_torques, d_bonds, _grooving, _use_oxDNA2_FENE, _use_mbf, _mbf_xmax,
                 _mbf_finf, _update_st, _d_st);

        if (_read_par) {
            if (_d_aff_eqdist == nullptr || _d_aff_gamma == nullptr ||
                _d_affected_indx == nullptr || _d_affected == nullptr ||
                _d_anm_v9_physical_to_slot == nullptr) {
                throw oxDNAException("CUDA ANM CSR pointers missing with active parfile");
            }
        protein_forces_edge_bonded
        <<<_launch_cfg.blocks, _launch_cfg.threads_per_block>>>
                (d_poss, d_orientations, d_forces, d_torques, d_bonds, _grooving, _use_oxDNA2_FENE, _use_mbf, _mbf_xmax,
                 _mbf_finf, d_box, _update_st, _d_st, _d_aff_eqdist, _d_aff_gamma, _d_affected_indx, _d_affected,
                     _d_anm_v9_physical_to_slot);
        }
        // CPU-equivalent DNACT bending/torsion contribution. This is
        // deliberately separate from the ANM spring kernel.
        if(_angular) {
            if(!_read_par || _d_ang_params == nullptr ||
               _d_affected_indx == nullptr || _d_affected == nullptr) {
                throw oxDNAException(
                    "DNACT CUDA angular candidate requires a validated ANM parameter file"
                );
            }
            dnact_protein_excluded_pairs
                <<<_launch_cfg.blocks, _launch_cfg.threads_per_block>>>(
                    d_poss, d_forces, d_bonds, d_box,
                    _d_protein_charge, _d_protein_group,
                    this->_protein_protein_electrostatics,
                    this->_exclude_intragroup_electrostatics,
                    _cuda_protein_protein_dh_scale,
                    _d_affected_indx, _d_affected,
                    _update_st, _d_st
                );
            dnact_protein_angular_forces_candidate
                <<<_launch_cfg.blocks, _launch_cfg.threads_per_block>>>(
                    d_poss, d_orientations, d_forces, d_torques, d_box,
                    _d_ang_params, _d_ang_kbkt,
                    _d_affected_indx, _d_affected,
                    _update_st, _d_st,
                    _d_anm_v9_physical_to_slot
                );
        }
    } else throw oxDNAException("Edge Approach is only implemented for DNANM Interaction using CUDA approach. Please add use_edge = 1 to your input file.");


//            if(_angular){
//                dnanm_forces_edge_bonded_angular
//                        <<< this->_launch_cfg.blocks, this->_launch_cfg.threads_per_block >>>
//                (d_poss, d_orientations, d_forces, d_torques, d_bonds, this->_grooving, _use_oxDNA2_FENE, this->_use_mbf, this->_mbf_xmax, this->_mbf_finf, d_box, _d_aff_eqdist, _d_aff_gamma, _d_ang_params, _d_ang_kbkt, _d_affected_indx, _d_affected);
//
//            } else {
//                dnanm_forces_edge_bonded
//                        <<< this->_launch_cfg.blocks, this->_launch_cfg.threads_per_block >>>
//                (d_poss, d_orientations, d_forces, d_torques, d_bonds, this->_grooving, _use_oxDNA2_FENE, this->_use_mbf, this->_mbf_xmax, this->_mbf_finf, d_box, _d_aff_eqdist, _d_aff_gamma, _d_affected_indx, _d_affected);
//            }

//    } else throw oxDNAException("Must Use with Lists to run simulation");
}

void CUDADNANMInteraction::_on_T_update() {
    // The backend's neighbour matrix was allocated for the initial cutoff.
    // Rebuild constants and ANM buffers, but NEVER silently exceed its radius.
    if(!_anm_v9_list_cutoff_recorded) throw oxDNAException("ANM cutoff not initialized");
    cuda_init(_N);
    if(get_cuda_rcut() > _anm_v9_initial_list_cutoff + (c_number) 1.e-6)
        throw oxDNAException("ANM temperature/salt update enlarges cutoff: restart to rebuild CUDA neighbour lists");
}

void CUDADNANMInteraction::_init_strand_ends(LR_bonds *d_bonds) {
    CUDA_SAFE_CALL(GpuUtils::LR_cudaMalloc<int>(&_d_is_strand_end, sizeof(int) * _N));
    dna_init_DNA_strand_ends<<<_launch_cfg.blocks, _launch_cfg.threads_per_block>>>(_d_is_strand_end, d_bonds, _N);
}
