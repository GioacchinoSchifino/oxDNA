/*
 * DNANMInteraction.cpp
 *
 *  Created on: Feb 2026
 *  Author: jonah,subhajit
 *  Inherits from the DNA2Interaction Class
 *  Uses DNA2 Model and ANM Protein Model
 */


#include "DNANMInteraction.h"

#include "../Particles/DNANucleotide.h"
#include "../Particles/ANMParticle.h"
#include "../Utilities/Utils.h"

#include <sstream>
#include <fstream>
#include <algorithm>
#include <cmath>
#include <limits>
#include <cctype>
#include <cerrno>
#include <cstdlib>
#include <map>
#include <set>
#include <utility>
#include <cstring>


namespace {

std::string trim_copy(const std::string &value) {
    std::size_t first = 0;
    while(first < value.size() &&
          std::isspace(static_cast<unsigned char>(value[first]))) {
        first++;
    }

    std::size_t last = value.size();
    while(last > first &&
          std::isspace(static_cast<unsigned char>(value[last - 1]))) {
        last--;
    }

    return value.substr(first, last - first);
}

bool starts_with(const std::string &value, const std::string &prefix) {
    return value.size() >= prefix.size() &&
           value.compare(0, prefix.size(), prefix) == 0;
}

double parse_finite_double_text(
    const std::string &value,
    const std::string &what
) {
    if(value.empty()) {
        throw oxDNAException(
            "%s is empty",
            what.c_str()
        );
    }

    // oxDNA Release builds use -ffast-math, which includes
    // -ffinite-math-only. Reject NaN/Inf spellings lexically before
    // conversion rather than relying on floating-point classification.
    std::string lower_value(value);

    std::transform(
        lower_value.begin(),
        lower_value.end(),
        lower_value.begin(),
        [](unsigned char c) {
            return static_cast<char>(std::tolower(c));
        }
    );

    std::size_t token_start = 0;

    if(!lower_value.empty() &&
       (lower_value[0] == '+' || lower_value[0] == '-')) {
        token_start = 1;
    }

    const std::string unsigned_value =
        lower_value.substr(token_start);

    if(unsigned_value == "inf" ||
       unsigned_value == "infinity" ||
       starts_with(unsigned_value, "nan")) {
        throw oxDNAException(
            "%s is out of range or non-finite (`%s')",
            what.c_str(),
            value.c_str()
        );
    }

    errno = 0;
    char *end = nullptr;
    const char *begin = value.c_str();
    const double parsed = std::strtod(begin, &end);

    if(end == begin) {
        throw oxDNAException(
            "%s has invalid value `%s'",
            what.c_str(),
            value.c_str()
        );
    }

    while(end != nullptr &&
          *end != '\0' &&
          std::isspace(static_cast<unsigned char>(*end))) {
        end++;
    }

    if(end == nullptr || *end != '\0') {
        throw oxDNAException(
            "%s has invalid trailing characters in value `%s'",
            what.c_str(),
            value.c_str()
        );
    }

    if(errno == ERANGE) {
        throw oxDNAException(
            "%s is out of range or non-finite (`%s')",
            what.c_str(),
            value.c_str()
        );
    }

    return parsed;
}

number checked_number_product(
    number lhs,
    number rhs,
    const char *what
) {
    const number lhs_abs = std::fabs(lhs);
    const number rhs_abs = std::fabs(rhs);
    const number limit = std::numeric_limits<number>::max();

    if(lhs_abs != (number) 0.0 &&
       rhs_abs > limit / lhs_abs) {
        throw oxDNAException(
            "%s overflows the compiled CPU numerical type",
            what
        );
    }

    return lhs * rhs;
}

number checked_reduced_protein_charge(
    number protein_charge_scale,
    double charge_e,
    number dna_effective_charge_e,
    int local_index
) {
    const double scale = static_cast<double>(protein_charge_scale);
    const double dna_charge = static_cast<double>(dna_effective_charge_e);
    const double product_limit = std::numeric_limits<double>::max();

    const double scale_abs = std::fabs(scale);
    const double charge_abs = std::fabs(charge_e);

    if(scale_abs != 0.0 &&
       charge_abs > product_limit / scale_abs) {
        throw oxDNAException(
            "Reduced protein charge for local index %d overflows while multiplying protein_charge_scale by charge_e",
            local_index
        );
    }

    const double numerator = scale * charge_e;
    const double numerator_abs = std::fabs(numerator);

    // dna_effective_charge_e is required to be positive. Division can
    // overflow only when the divisor is below one; the product on the
    // right remains <= DBL_MAX and therefore cannot overflow here.
    if(dna_charge < 1.0 &&
       numerator_abs > product_limit * dna_charge) {
        throw oxDNAException(
            "Reduced protein charge for local index %d overflows while dividing by dna_effective_charge_e",
            local_index
        );
    }

    const double reduced = numerator / dna_charge;
    const double number_limit =
        static_cast<double>(std::numeric_limits<number>::max());

    if(reduced < -number_limit || reduced > number_limit) {
        throw oxDNAException(
            "Reduced protein charge for local index %d is outside the compiled CPU numerical range",
            local_index
        );
    }

    return static_cast<number>(reduced);
}

bool parse_optional_number_strict(
    input_file &inp,
    const char *key,
    number &dest
) {
    std::string raw;

    if(getInputString(&inp, key, raw, 0) != KEY_FOUND) {
        return false;
    }

    const std::string value = trim_copy(raw);
    const std::string what =
        std::string("Numeric input key `") + key + "'";

    const double parsed =
        parse_finite_double_text(value, what);

    const double limit =
        static_cast<double>(std::numeric_limits<number>::max());

    if(parsed < -limit || parsed > limit) {
        throw oxDNAException(
            "Numeric input key `%s' is outside the compiled CPU numerical range (`%s')",
            key,
            value.c_str()
        );
    }

    dest = static_cast<number>(parsed);
    return true;
}

} // namespace


DNANMInteraction::DNANMInteraction(bool btp) : DNA2Interaction() {

    _analytic_cross_stacking = false;
    _angular = btp;
    //Protein Methods Function Pointers
    ADD_INTERACTION_TO_MAP(SPRING, _protein_spring);
    ADD_INTERACTION_TO_MAP(PRO_EXC_VOL, _protein_exc_volume);
    if(_angular)
        ADD_INTERACTION_TO_MAP(PRO_ANG_POT, _protein_ang_pot);

    //Protein-DNA Function Pointers
    ADD_INTERACTION_TO_MAP(PRO_DNA_EXC_VOL, _protein_dna_exc_volume);


    //DNA Methods Function Pointers
    ADD_INTERACTION_TO_MAP(BACKBONE, _backbone);
    ADD_INTERACTION_TO_MAP(COAXIAL_STACKING, _coaxial_stacking);
    ADD_INTERACTION_TO_MAP(CROSS_STACKING, _cross_stacking);
    ADD_INTERACTION_TO_MAP(BONDED_EXCLUDED_VOLUME, _bonded_excluded_volume);
    ADD_INTERACTION_TO_MAP(STACKING, _stacking);
    ADD_INTERACTION_TO_MAP(HYDROGEN_BONDING, _hydrogen_bonding);
    ADD_INTERACTION_TO_MAP(NONBONDED_EXCLUDED_VOLUME, _nonbonded_excluded_volume);
    ADD_INTERACTION_TO_MAP(DEBYE_HUCKEL, _debye_huckel);

    _parameter_kbkt = false;

    _protein_dna_electrostatics = false;
    _protein_protein_electrostatics = false;
    _exclude_intragroup_electrostatics = true;

    _protein_charge_scale = (number) 1.0;
    _protein_dna_dh_scale = (number) 1.0;
    _protein_protein_dh_scale = (number) 1.0;

    // Standard oxDNA2 effective backbone charge reference.
    _dna_effective_charge_e = (number) 0.815;

    _protein_charge_format_version = 0;

    masses = {{0, 1.f}, {1, 1.f}, {2, 1.f}, {3, 1.f}, {5, 1.f}, {6, 1.f}, //default mass is 1 for everything
              {7, 1.f}, {8, 1.f}, {9, 1.f}, {10, 1.f},{11, 1.f}, {12, 1.f},
              {13, 1.f}, {14, 1.f}, {15, 1.f}, {16, 1.f}, {17, 1.f}, {18, 1.f},
              {19, 1.f}, {20, 1.f}, {21, 1.f}, {22, 1.f}, {23, 1.f}, {24, 1.f}};
}


// Gate 4A opt-in analytic cross-stacking, matching the native GPU f4
// evaluation, without changing the base DNA2 and non-cross f4 pathways.
number DNANMInteraction::_custom_f4(number cost, int i) {
    if(_analytic_cross_stacking &&
       i >= CRST_F4_THETA1 && i <= CRST_F4_THETA7) {
        return this->_fakef4(cost, (void *)&i);
    }
    return DNA2Interaction::_custom_f4(cost, i);
}

number DNANMInteraction::_custom_f4D(number cost, int i) {
    if(_analytic_cross_stacking &&
       i >= CRST_F4_THETA1 && i <= CRST_F4_THETA7) {
        return this->_fakef4D(cost, (void *)&i);
    }
    return DNA2Interaction::_custom_f4D(cost, i);
}

void DNANMInteraction::get_settings(input_file &inp){
    // C09: fail closed for both barostat NPT and Monte Carlo volume moves.
    bool npt_barostat = false;
    getInputBool(&inp, "use_barostat", &npt_barostat, 0);
    std::string sim_type, ensemble;
    getInputString(&inp, "sim_type", sim_type, 0);
    getInputString(&inp, "ensemble", ensemble, 0);
    std::transform(sim_type.begin(), sim_type.end(), sim_type.begin(),
                   [](unsigned char c){ return static_cast<char>(std::tolower(c)); });
    std::transform(ensemble.begin(), ensemble.end(), ensemble.begin(),
                   [](unsigned char c){ return static_cast<char>(std::tolower(c)); });
    if(npt_barostat || (sim_type == "mc" && ensemble == "npt"))
        throw oxDNAException("DNANM/DNACT NPT is not qualified in v1.1.0");

    // Validate text before legacy atof parsing: Release uses -ffast-math,
    // so std::isfinite alone cannot enforce rejection of NaN/Inf inputs.
    for(const char *key : {"salt_concentration", "dh_lambda", "dh_strength", "debye_huckel_rhigh"}) {
        number requested = (number) 0.;
        if(parse_optional_number_strict(inp, key, requested) &&
           (requested <= (number) 0. ||
            requested > (number) std::numeric_limits<float>::max()))
            throw oxDNAException("DNANM/DNACT numeric key %s must be positive and in CUDA float range", key);
    }
    std::string raw_temperature;
    if(getInputString(&inp, "T", raw_temperature, 0) == KEY_FOUND) {
        raw_temperature = trim_copy(raw_temperature);
        if(!raw_temperature.empty() &&
           (raw_temperature.back() == 'K' || raw_temperature.back() == 'k' ||
            raw_temperature.back() == 'C' || raw_temperature.back() == 'c'))
            raw_temperature.pop_back();
        const double requested_T = parse_finite_double_text(trim_copy(raw_temperature), "temperature T");
        if(requested_T < -(double) std::numeric_limits<float>::max() ||
           requested_T > (double) std::numeric_limits<float>::max())
            throw oxDNAException("DNANM/DNACT temperature outside CUDA float range");
    }
    this->DNA2Interaction::get_settings(inp);
    // DNANM/DNACT keeps the qualified DNA Hamiltonian intact. The inherited
    // switches are parsed by DNA2Interaction but are not implemented by the
    // hybrid CPU/CUDA force paths, so a disabled term must fail explicitly.
    const std::pair<const char *, bool> dna_terms[] = {
        {"DNA_enable_backbone", _enable_backbone},
        {"DNA_enable_bonded_excluded_volume", _enable_bonded_excluded_volume},
        {"DNA_enable_stacking", _enable_stacking},
        {"DNA_enable_nonbonded_excluded_volume", _enable_nonbonded_excluded_volume},
        {"DNA_enable_hydrogen_bonding", _enable_hydrogen_bonding},
        {"DNA_enable_cross_stacking", _enable_cross_stacking},
        {"DNA_enable_coaxial_stacking", _enable_coaxial_stacking},
        {"DNA_enable_debye_huckel", _enable_debye_huckel}
    };
    for(const auto &term : dna_terms) {
        if(!term.second)
            throw oxDNAException("DNANM/DNACT does not support %s = false; all inherited DNA terms must remain enabled", term.first);
    }
    // C07: check native DH parameters as well as the added charge-file keys.
    // Reject zero strength until a zero-strength analytical path is validated.
    const double T = static_cast<double>(_T);
    const double salt = static_cast<double>(_salt_concentration);
    const double lambda0 = static_cast<double>(_debye_huckel_lambdafactor);
    const double strength = static_cast<double>(_debye_huckel_prefactor);
    const double rhigh = static_cast<double>(_debye_huckel_RHIGH);
    if(!std::isfinite(T) || T <= 0. || !std::isfinite(salt) || salt <= 0. ||
       !std::isfinite(lambda0) || lambda0 <= 0. ||
       !std::isfinite(strength) || strength <= 0. ||
       !std::isfinite(rhigh) || rhigh <= 0.)
        throw oxDNAException("Nonfinite or nonpositive DNANM/DNACT Debye-Huckel parameter");
    const double lambda = lambda0 * std::sqrt(T / 0.1) / std::sqrt(salt);
    if(!std::isfinite(lambda) || lambda <= 0.)
        throw oxDNAException("Invalid DNANM/DNACT Debye-Huckel screening length");
    // Defaults to false; must be explicitly requested for the diagnostic.
    getInputBool(&inp, "dnanm_analytic_cross_stacking", &_analytic_cross_stacking, 0);

    // Require an explicit parameter policy for every hybrid input. DNANM may
    // select "none"; DNACT needs a file defining its angular connectivity.
    // Read into std::string before copying into the legacy CUDA-shared buffer.
    std::string parameter_file;
    getInputString(&inp, "parfile", parameter_file, 1);
    parameter_file = trim_copy(parameter_file);
    if(parameter_file.empty() || parameter_file.find('\0') != std::string::npos ||
       parameter_file.size() >= sizeof(_parameterfile))
        throw oxDNAException("Invalid parfile: expected a nonempty path shorter than %zu bytes, or explicit none for DNANM", sizeof(_parameterfile));
    if(_angular && parameter_file == "none")
        throw oxDNAException("DNACT requires a validated ANM parameter file; parfile = none is supported only by DNANM");
    std::copy(parameter_file.begin(), parameter_file.end(), _parameterfile);
    _parameterfile[parameter_file.size()] = '\0';

    if(_angular){
        number kb_tmp = (number) -1.;
        number kt_tmp = (number) -1.;
        parse_optional_number_strict(inp, "bending_k", kb_tmp);
        parse_optional_number_strict(inp, "torsion_k", kt_tmp);
        const number stiffness_limit = (number) std::numeric_limits<float>::max();
        if(kb_tmp < -stiffness_limit || kb_tmp > stiffness_limit ||
           kt_tmp < -stiffness_limit || kt_tmp > stiffness_limit)
            throw oxDNAException("DNACT global stiffness outside CUDA float range");
        _k_bend = kb_tmp;
        _k_tor = kt_tmp;

        if((_k_bend < 0 && kt_tmp >= 0) || (_k_tor < 0 && _k_bend >= 0)){
            throw oxDNAException("Bending & Torsion Constants Must BOTH be set globally or in parameter file");
        } else if(_k_bend >= 0 && _k_tor >= 0){
            OX_LOG(Logger::LOG_INFO, "Using Global kb/kt values set in Input File");
        } else if(_k_bend < 0 && _k_tor < 0){
            OX_LOG(Logger::LOG_INFO, "No Global kb/kt values set in Input File, Attempting to read from Parameter File");
            _parameter_kbkt = true;
        }
    }

    if(getInputString(&inp, "massfile", _massfile, 0) == KEY_NOT_FOUND) { // optional unit-mass table
        OX_LOG(Logger::LOG_INFO, "Using Default Masses"); // declared in constructor
    } else {
        OX_LOG(Logger::LOG_INFO, "Validating Provided Unit-Mass File");
        load_massfile(_massfile);
    }
    getInputBool(
        &inp,
        "protein_dna_electrostatics",
        &_protein_dna_electrostatics,
        0
    );



    getInputBool(
        &inp,
        "exclude_intragroup_electrostatics",
        &_exclude_intragroup_electrostatics,
        0
    );

    parse_optional_number_strict(
        inp,
        "protein_charge_scale",
        _protein_charge_scale
    );

    parse_optional_number_strict(
        inp,
        "protein_dna_dh_scale",
        _protein_dna_dh_scale
    );

    parse_optional_number_strict(
        inp,
        "protein_protein_dh_scale",
        _protein_protein_dh_scale
    );

    parse_optional_number_strict(
        inp,
        "dna_effective_charge_e",
        _dna_effective_charge_e
    );

    if(_dna_effective_charge_e <= (number) 0.0) {
        throw oxDNAException(
            "dna_effective_charge_e must be positive"
        );
    }

    if(_protein_charge_scale < (number) 0.0) {
        throw oxDNAException(
            "protein_charge_scale must be non-negative"
        );
    }

    if(_protein_dna_dh_scale < (number) 0.0) {
        throw oxDNAException(
            "protein_dna_dh_scale must be non-negative"
        );
    }

    if(_protein_protein_dh_scale < (number) 0.0) {
        throw oxDNAException(
            "protein_protein_dh_scale must be non-negative"
        );
    }

    if(_protein_dna_electrostatics) {
        ADD_INTERACTION_TO_MAP(
            PRO_DNA_DH,
            _protein_dna_debye_huckel
        );
    }



    getInputBool(&inp, "protein_protein_electrostatics", &_protein_protein_electrostatics, 0);
    if(_protein_protein_electrostatics) {
        ADD_INTERACTION_TO_MAP(PRO_PRO_DH, _protein_protein_debye_huckel);
    }
    if(_protein_dna_electrostatics ||
       _protein_protein_electrostatics) {

        if(getInputString(
               &inp,
               "protein_charge_file",
               _protein_charge_file,
               0
           ) != KEY_FOUND) {

            throw oxDNAException(
                "protein_charge_file is required when protein electrostatics are enabled"
            );
        }

        OX_LOG(
            Logger::LOG_INFO,
            "Protein electrostatics enabled: PD=%d PP=%d, "
            "charge scale=%g, PD scale=%g, PP scale=%g, "
            "oxDNA2 charge-normalisation reference=%g e",
            (int) _protein_dna_electrostatics,
            (int) _protein_protein_electrostatics,
            _protein_charge_scale,
            _protein_dna_dh_scale,
            _protein_protein_dh_scale,
            _dna_effective_charge_e
        );
    }


}


void DNANMInteraction::read_parameter_file(std::vector<BaseParticle*> &particles) {
    // Keep the legacy DNANM/DNACT spring Hamiltonian and parameter format.
    // Parse one physical record per line: a malformed row must never silently
    // truncate the spring network or consume columns from the following row.
    if(std::strcmp(_parameterfile, "none") == 0 || _parameterfile[0] == '\0') {
        OX_LOG(Logger::LOG_INFO, "Parfile: NONE, No protein parameters were filled");
        return;
    }

    std::ifstream parameters(_parameterfile);
    if(!parameters.is_open())
        throw oxDNAException("ParameterFile Could Not Be Opened");

    std::string row;
    if(!std::getline(parameters, row))
        throw oxDNAException("ANM parameter file has no protein-count header");
    std::istringstream header(row);
    int N = -1;
    std::string remainder;
    if(!(header >> N) || N != npro || N <= 0 ||
       ((header >> remainder) && remainder[0] != '#'))
        throw oxDNAException("Invalid ANM parameter header: expected npro=%d protein beads", npro);

    int spring_connection_num = 0;
    int line_no = 1;
    std::set<std::pair<int,int>> anm_seen_edges;
    while(std::getline(parameters, row)) {
        ++line_no;
        const std::string trimmed = trim_copy(row);
        if(trimmed.empty() || trimmed[0] == '#') continue;
        std::istringstream fields(trimmed);

        int key1 = -1, key2 = -1;
        double dist = 0., potential = 0.;
        char potswitch = '\0';
        if(!(fields >> key1 >> key2 >> dist >> potswitch >> potential))
            throw oxDNAException("Truncated or malformed ANM spring on parameter line %d", line_no);
        if(key1 < 0 ||  key2 < 0 ||
           key1 >= static_cast<int>(particles.size()) ||
           key2 >= static_cast<int>(particles.size()))
            throw oxDNAException("Invalid Particle ID %d in Parameter File (line %d)",
                                 (key1 < 0 ||  key1 >= static_cast<int>(particles.size())) ? key1 : key2,
                                 line_no);
        if(dynamic_cast<ANMParticle*>(particles[key1]) == nullptr ||
           dynamic_cast<ANMParticle*>(particles[key2]) == nullptr)
            throw oxDNAException("ANM spring endpoint is not a protein bead on parameter line %d", line_no);
        if(!std::isfinite(dist) || dist <= 0. || !std::isfinite(potential) || potential < 0.)
            throw oxDNAException("Nonfinite or invalid ANM distance/stiffness on parameter line %d", line_no);
        if(potswitch != 's')
            throw oxDNAException("Unsupported ANM potential type on parameter line %d", line_no);
        if(key1 == key2)
            throw oxDNAException("ANM self-spring on parameter line %d", line_no);
        // Spring lookup and bonded-neighbour construction use ordered keys.
        // Reject reversed records instead of silently omitting an ANM bond.
        if(key1 > key2)
            throw oxDNAException("ANM spring endpoints must be in ascending order on line %d", line_no);
        if(!anm_seen_edges.insert(std::minmax(key1, key2)).second)
            throw oxDNAException("Duplicate ANM spring edge on parameter line %d", line_no);

        // DNACT supports four angular cosines on consecutive spring edges,
        // and optionally two pairwise stiffnesses, in the original order.
        if(_angular && key2 - key1 == 1) {
            double a0, b0, c0, d0;
            if(!(fields >> a0 >> b0 >> c0 >> d0))
                throw oxDNAException("Missing DNACT angular parameters on line %d", line_no);
            if(!std::isfinite(a0) || !std::isfinite(b0) ||
               !std::isfinite(c0) || !std::isfinite(d0) ||
               std::min({a0,b0,c0,d0}) < -1. ||
               std::max({a0,b0,c0,d0}) > 1.)
                throw oxDNAException("Invalid DNACT angular cosine on line %d", line_no);
            if(_parameter_kbkt) {
                double kb = 0., kt = 0.;
                if(!(fields >> kb >> kt) || !std::isfinite(kb) || !std::isfinite(kt) ||
                   kb < 0. || kt < 0.)
                    throw oxDNAException("Missing or invalid DNACT pairwise stiffness on line %d", line_no);
                _ang_stiff[key1] = {kb, kt};
            }
            _ang_vals[key1] = {a0, b0, c0, d0};
        }
        if(fields >> remainder) {
            if(remainder[0] != '#')
                throw oxDNAException("Unexpected ANM parameter fields on line %d", line_no);
        }

        if(key1 < key2)
            static_cast<ANMParticle*>(particles[key1])->add_bonded_neighbor(particles[key2]);
        ++spring_connection_num;
        const std::pair<int,int> edge(key1, key2);
        _rknot[edge] = dist;
        _potential[edge] = {potswitch, potential};
    }
    if(parameters.bad())
        throw oxDNAException("I/O error reading ANM parameters");

    // Preserve the upstream distinction between DNANM and DNACT parameter
    // conventions and its legacy single-spring consistency checks.
    if(_angular) {
        if(spring_connection_num == 1 && N > 2 && _parameter_kbkt)
            throw oxDNAException("Invalid Parameter File Format, pairwise kb and kt values incorrectly formatted or missing");
        if(spring_connection_num == 1 && N > 2 && !_parameter_kbkt)
            throw oxDNAException("Invalid Parameter File Format, global kb and kt values have been set in Inputfile");
    } else if(spring_connection_num == 1 && N > 2) {
        throw oxDNAException("Invalid Parameter File Format, cannot use a DNACT Parameter File");
    }
}
void DNANMInteraction::check_input_sanity(std::vector<BaseParticle*> &particles){
    //this->DNA2Interaction::check_input_sanity(particles,N);
    //Need to make own function that checks the input sanity
}


void DNANMInteraction::allocate_particles(std::vector<BaseParticle*> &particles) {
    if (ndna==0 || ndnas==0) {
        OX_LOG(Logger::LOG_INFO, "No DNA Particles Specified, Continuing with just Protein Particles");
        for (int i = 0; i < npro; i++) particles[i] = new ANMParticle(_angular);

    } else if (npro == 0) {
        OX_LOG(Logger::LOG_INFO, "No Protein Particles Specified, Continuing with just DNA Particles");
        for (int i = 0; i < ndna; i++) particles[i] = new DNANucleotide(this->_grooving);

    } else {
        if (_firststrand > 0){
            for (int i = 0; i < ndna; i++) particles[i] = new DNANucleotide(this->_grooving);
            // Protein
            for (uint i = ndna; i < particles.size(); i++) particles[i] = new ANMParticle(_angular);
        } else {
            // Protein
            for (int i = 0; i < npro; i++) particles[i] = new ANMParticle(_angular);

            for (uint i = npro; i < particles.size(); i++) particles[i] = new DNANucleotide(this->_grooving);
        }
    }
}


void DNANMInteraction::read_topology(int *N_strands, std::vector<BaseParticle*> &particles) {
    // The legacy hybrid topology allocates particle types by contiguous species
    // blocks. Reject malformed headers and rows before dereferencing particles.
    std::ifstream topology(this->_topology_filename);
    if(!topology.good())
        throw oxDNAException("Can't read topology file '%s'", this->_topology_filename);

    std::string line;
    if(!std::getline(topology, line))
        throw oxDNAException("Missing DNANM topology header");

    int my_N = -1, my_N_strands = -1;
    std::stringstream head(line);
    std::string extra;
    if(!(head >> my_N >> my_N_strands >> ndna >> npro >> ndnas) ||
       (head >> extra) || my_N <= 0 || my_N_strands <= 0 ||
       my_N_strands > my_N || ndna < 0 || npro < 0 || ndnas < 0 ||
       ndnas > my_N_strands || ndna > my_N || npro > my_N ||
       ndna != my_N - npro ||
       ((ndna == 0) != (ndnas == 0))) {
        throw oxDNAException("Invalid DNANM topology header: expected '<N> <N_strands> <ndna> <npro> <ndnas>' with N=ndna+npro");
    }
    if(static_cast<int>(particles.size()) != my_N)
        throw oxDNAException("DNANM topology contains %d particles but configuration expects %d", my_N, (int)particles.size());

    if(_protein_dna_electrostatics || _protein_protein_electrostatics) {
        _protein_local_to_global.clear();
        _protein_topology_strand.clear();
        _protein_residue_code.clear();
    }

    int i = 0;
    bool allocated = false;
    bool protein_first = false;
    int line_no = 1;
    while(std::getline(topology, line)) {
        ++line_no;
        const std::string trimmed = trim_copy(line);
        if(trimmed.empty() || trimmed[0] == '#') continue;
        if(i >= my_N)
            throw oxDNAException("Too many particles in DNANM topology (expected %d)", my_N);

        std::stringstream ss(trimmed);
        int strand = 0;
        if(!(ss >> strand) || strand == 0)
            throw oxDNAException("Invalid strand identifier on DNANM topology line %d", line_no);

        if(!allocated) {
            _firststrand = strand;
            protein_first = strand < 0;
            allocate_particles(particles);
            for(int j = 0; j < my_N; ++j) {
                if(particles[j] == nullptr)
                    throw oxDNAException("DNANM particle allocation failed at index %d", j);
                particles[j]->index = j;
                particles[j]->type = P_INVALID;
                particles[j]->strand_id = 0;
            }
            allocated = true;
        }

        // The original DNANM representation supports exactly one contiguous
        // block per species (DNA-first OR protein-first). Validate that here.
        const bool slot_is_protein = protein_first ? (i < npro) : (i >= ndna);
        if((strand < 0) != slot_is_protein)
            throw oxDNAException("DNANM topology line %d violates the contiguous protein/DNA block layout", line_no);
        // Strand identifiers are signed: protein chains use -1, -2, ...
        // and DNA strands use 1, 2, ... . Validate before std::abs(strand)
        // and before constructing particle strand IDs; abs(INT_MIN) is UB.
        if(strand < 0) {
            if(strand == std::numeric_limits<int>::min() ||
               -strand > my_N_strands - ndnas)
                throw oxDNAException("Protein strand identifier outside topology header range on line %d", line_no);
        } else if(strand > ndnas) {
            throw oxDNAException("DNA strand identifier outside topology header range on line %d", line_no);
        }

        if(strand < 0) {
            auto *p = dynamic_cast<ANMParticle *>(particles[i]);
            if(!p) throw oxDNAException("Expected protein bead on topology line %d", line_no);
            std::string aminoacid;
            int nside = -1, cside = -1;
            if(!(ss >> aminoacid >> nside >> cside) || aminoacid.size() != 1)
                throw oxDNAException("Malformed protein record on topology line %d", line_no);

            auto valid_protein_link = [&](int target) {
                if(target == -1) return;
                if(target < 0 || target >= my_N ||
                   (protein_first ? (target >= npro) : (target < ndna)))
                    throw oxDNAException("Invalid protein neighbour %d on topology line %d", target, line_no);
            };
            valid_protein_link(nside);
            valid_protein_link(cside);
            if(_angular) {
                if(nside >= 0) p->n3 = particles[nside];
                if(cside >= 0) p->n5 = particles[cside];
            }
            std::set<int> myneighs;
            if(nside >= 0) myneighs.insert(nside);
            if(cside >= 0) myneighs.insert(cside);
            std::string field;
            while(ss >> field) {
                int x = -1;
                std::stringstream fs(field);
                std::string junk;
                if(!(fs >> x) || (fs >> junk))
                    throw oxDNAException("Noninteger protein neighbour on topology line %d", line_no);
                valid_protein_link(x);
                if(x < 0)
                    throw oxDNAException("Unexpected negative protein neighbour on line %d", line_no);
                myneighs.insert(x);
            }

            const char code = static_cast<char>(std::toupper(static_cast<unsigned char>(aminoacid[0])));
            p->type = Utils::decode_aa(code);
            p->btype = p->type;
            // decode_aa() reports unknown amino acids as A_INVALID (0),
            // not P_INVALID (-1); the latter is the DNA/particle sentinel.
            // In particular, type 0 has a default mass and would otherwise
            // be misclassified as DNA in get_id(), causing undefined access.
            if(p->type == A_INVALID || p->type == P_INVALID ||
               masses.find(p->type) == masses.end() || masses[p->type] <= 0)
                throw oxDNAException("Invalid protein residue or missing massfile type on DNANM topology line %d", line_no);
            if(masses[p->type] != (number) 1.0)
                throw oxDNAException("DNANM/DNACT supports unit protein masses only (topology line %d)", line_no);
            p->mass = masses[p->type];
            p->invmass = 1.f / p->mass;
            p->strand_id = std::abs(strand) + ndnas - 1;
            p->index = i;
            if(_protein_dna_electrostatics || _protein_protein_electrostatics) {
                _protein_local_to_global.push_back(i);
                _protein_topology_strand.push_back(strand);
                _protein_residue_code.push_back(code);
            }
        } else {
            auto *p = dynamic_cast<DNANucleotide *>(particles[i]);
            if(!p) throw oxDNAException("Expected DNA nucleotide on topology line %d", line_no);
            std::string base;
            int tmpn3 = -1, tmpn5 = -1;
            if(!(ss >> base >> tmpn3 >> tmpn5) || base.size() != 1 || (ss >> extra))
                throw oxDNAException("Malformed DNA record on topology line %d", line_no);
            auto valid_dna_link = [&](int target) {
                if(target == -1) return;
                if(target < 0 || target >= my_N ||
                   (protein_first ? (target < npro) : (target >= ndna)))
                    throw oxDNAException("Invalid DNA neighbour %d on topology line %d", target, line_no);
            };
            valid_dna_link(tmpn3);
            valid_dna_link(tmpn5);
            p->n3 = tmpn3 < 0 ? P_VIRTUAL : particles[tmpn3];
            p->n5 = tmpn5 < 0 ? P_VIRTUAL : particles[tmpn5];
            p->strand_id = strand - 1;
            p->type = Utils::decode_base(base[0]);
            p->btype = p->type;
            if(p->type == P_INVALID)
                throw oxDNAException("Invalid DNA base '%c' on topology line %d", base[0], line_no);
            p->index = i;
            if(p->n3 != P_VIRTUAL) p->affected.emplace_back(ParticlePair(p->n3, p));
            if(p->n5 != P_VIRTUAL) p->affected.emplace_back(ParticlePair(p, p->n5));
        }
        ++i;
    }
    if(i != my_N)
        throw oxDNAException("Not enough DNANM particles: found %d, expected %d", i, my_N);

    *N_strands = my_N_strands;
    read_parameter_file(particles);
    if(_protein_dna_electrostatics || _protein_protein_electrostatics) {
        if(static_cast<int>(_protein_local_to_global.size()) != npro)
            throw oxDNAException("Mismatch between protein topology rows and header count");
        const int anm_expected_offset = protein_first ? 0 : ndna;
        for(int li = 0; li < npro; ++li)
            if(_protein_local_to_global[li] != anm_expected_offset + li)
                throw oxDNAException("ANM requires a contiguous block of protein beads in the topology");
        _load_protein_charge_file(my_N);
    }
}

void DNANMInteraction::_load_protein_charge_file(int total_particles) {

    if(!_protein_dna_electrostatics &&
       !_protein_protein_electrostatics) {
        return;
    }

    if(npro <= 0) {
        throw oxDNAException(
            "Protein electrostatics requested but topology contains no protein particles"
        );
    }

    if((int) _protein_local_to_global.size() != npro ||
       (int) _protein_topology_strand.size() != npro ||
       (int) _protein_residue_code.size() != npro) {

        throw oxDNAException(
            "Protein topology metadata were not initialised before loading '%s'",
            _protein_charge_file.c_str()
        );
    }

    _protein_charge_reduced.assign(
        total_particles,
        (number) 0.0
    );

    _protein_group.assign(
        total_particles,
        -1
    );

    std::vector<bool> seen(npro, false);

    std::ifstream charges(_protein_charge_file);

    if(!charges.good()) {
        throw oxDNAException(
            "Could not open protein charge file '%s'",
            _protein_charge_file.c_str()
        );
    }

    struct ChargeRecord {
        int local_index = -1;
        int topology_strand = 0;
        char residue = '\0';
        double charge_e = 0.0;
        int group_id = -1;
        int line_no = 0;
    };

    std::vector<ChargeRecord> records;
    std::map<std::string, std::string> metadata;

    int format_version = 0;
    bool saw_format_header = false;
    bool saw_data = false;

    std::string line;
    int line_no = 0;

    while(std::getline(charges, line)) {

        line_no++;

        std::string trimmed = trim_copy(line);

        if(trimmed.empty()) {
            continue;
        }

        if(trimmed[0] == '#') {

            std::string payload =
                trim_copy(trimmed.substr(1));

            if(starts_with(
                   payload,
                   "DNANM_ES_CHARGES"
               )) {

                if(saw_data) {
                    throw oxDNAException(
                        "Charge-format header appears after data on line %d of '%s'",
                        line_no,
                        _protein_charge_file.c_str()
                    );
                }

                std::stringstream hs(payload);
                std::string magic;
                std::string extra;

                if(saw_format_header) {
                    throw oxDNAException(
                        "Duplicate DNANM_ES_CHARGES header on line %d of '%s'",
                        line_no,
                        _protein_charge_file.c_str()
                    );
                }

                if(!(hs >> magic >> format_version) ||
                   magic != "DNANM_ES_CHARGES" ||
                   (hs >> extra)) {

                    throw oxDNAException(
                        "Malformed DNANM_ES_CHARGES header on line %d of '%s'",
                        line_no,
                        _protein_charge_file.c_str()
                    );
                }

                saw_format_header = true;

                if(format_version != 1) {
                    throw oxDNAException(
                        "Unsupported DNANM-ES charge-file version %d in '%s'",
                        format_version,
                        _protein_charge_file.c_str()
                    );
                }
            }
            else {
                const std::size_t eq =
                    payload.find('=');

                if(eq != std::string::npos) {
                    const std::string key =
                        trim_copy(payload.substr(0, eq));

                    const std::string value =
                        trim_copy(payload.substr(eq + 1));

                    if(metadata.find(key) != metadata.end()) {
                        throw oxDNAException(
                            "Duplicate metadata key '%s' on line %d of protein charge file '%s'",
                            key.c_str(),
                            line_no,
                            _protein_charge_file.c_str()
                        );
                    }

                    metadata.emplace(key, value);
                }
            }

            continue;
        }

        saw_data = true;

        const std::size_t hash =
            trimmed.find('#');

        if(hash != std::string::npos) {
            trimmed =
                trim_copy(trimmed.substr(0, hash));
        }

        if(trimmed.empty()) {
            continue;
        }

        ChargeRecord rec;
        rec.line_no = line_no;

        std::stringstream ss(trimmed);

        if(format_version == 1) {

            std::string residue;
            std::string charge_text;

            if(!(ss
                 >> rec.local_index
                 >> rec.topology_strand
                 >> residue
                 >> charge_text
                 >> rec.group_id) ||
               residue.size() != 1) {

                throw oxDNAException(
                    "Malformed version-1 record on line %d of protein charge file '%s'",
                    line_no,
                    _protein_charge_file.c_str()
                );
            }

            rec.residue =
                static_cast<char>(
                    std::toupper(
                        static_cast<unsigned char>(
                            residue[0]
                        )
                    )
                );

            rec.charge_e = parse_finite_double_text(
                charge_text,
                std::string("Protein charge on line ")
                    + std::to_string(line_no)
                    + " of '"
                    + _protein_charge_file
                    + "'"
            );
        }
        else {
            // Backward-compatible historical 3-column format:
            // local_index charge_e group_id
            std::string charge_text;

            if(!(ss
                 >> rec.local_index
                 >> charge_text
                 >> rec.group_id)) {

                throw oxDNAException(
                    "Malformed legacy record on line %d of protein charge file '%s'",
                    line_no,
                    _protein_charge_file.c_str()
                );
            }

            rec.charge_e = parse_finite_double_text(
                charge_text,
                std::string("Legacy protein charge on line ")
                    + std::to_string(line_no)
                    + " of '"
                    + _protein_charge_file
                    + "'"
            );
        }

        std::string extra;

        if(ss >> extra) {
            throw oxDNAException(
                "Unexpected extra field on line %d in protein charge file '%s'",
                line_no,
                _protein_charge_file.c_str()
            );
        }

        records.push_back(rec);
    }

    if(format_version == 1) {

        auto require_metadata =
            [&](const std::string &key)
            -> const std::string& {

            auto it = metadata.find(key);

            if(it == metadata.end()) {
                throw oxDNAException(
                    "Version-1 protein charge file '%s' is missing metadata '%s'",
                    _protein_charge_file.c_str(),
                    key.c_str()
                );
            }

            return it->second;
        };

        if(require_metadata("index_space") !=
           "protein_local") {

            throw oxDNAException(
                "Version-1 protein charge file '%s' must use index_space = protein_local",
                _protein_charge_file.c_str()
            );
        }

        if(require_metadata("index_base") != "0") {
            throw oxDNAException(
                "Version-1 protein charge file '%s' must use index_base = 0",
                _protein_charge_file.c_str()
            );
        }

        int declared_npro = -1;

        std::stringstream ns(
            require_metadata("n_protein_beads")
        );

        std::string npro_extra;

        if(!(ns >> declared_npro) ||
           (ns >> npro_extra)) {
            throw oxDNAException(
                "Malformed integer metadata 'n_protein_beads' in protein charge file '%s'",
                _protein_charge_file.c_str()
            );
        }

        if(declared_npro != npro) {

            throw oxDNAException(
                "Version-1 protein charge file '%s' declares n_protein_beads=%d but topology contains %d",
                _protein_charge_file.c_str(),
                declared_npro,
                npro
            );
        }

        auto model_it =
            metadata.find("charge_model");

        _protein_charge_model =
            (model_it == metadata.end())
            ? "unspecified"
            : model_it->second;

        auto source_it =
            metadata.find("charge_source");

        _protein_charge_source =
            (source_it == metadata.end())
            ? "unspecified"
            : source_it->second;
    }
    else {
        _protein_charge_model =
            "legacy-unspecified";

        _protein_charge_source =
            "legacy-unspecified";

        OX_LOG(
            Logger::LOG_WARNING,
            "Legacy three-column protein charge file detected: %s",
            _protein_charge_file.c_str()
        );
    }

    int nread = 0;

    number net_charge_e =
        (number) 0.0;

    number max_abs_charge_e =
        (number) 0.0;

    std::set<int> groups;

    for(const ChargeRecord &rec : records) {

        if(rec.local_index < 0 ||
           rec.local_index >= npro) {

            throw oxDNAException(
                "Protein local index %d out of range [0,%d) on line %d of '%s'",
                rec.local_index,
                npro,
                rec.line_no,
                _protein_charge_file.c_str()
            );
        }

        if(seen[rec.local_index]) {
            throw oxDNAException(
                "Duplicate protein local index %d in '%s'",
                rec.local_index,
                _protein_charge_file.c_str()
            );
        }

        if(rec.group_id < 0) {
            throw oxDNAException(
                "group_id must be non-negative on line %d of '%s'",
                rec.line_no,
                _protein_charge_file.c_str()
            );
        }

        if(format_version == 1) {

            if(rec.topology_strand !=
               _protein_topology_strand[
                   rec.local_index
               ]) {

                throw oxDNAException(
                    "Protein strand mismatch for local index %d on line %d of '%s': file=%d topology=%d",
                    rec.local_index,
                    rec.line_no,
                    _protein_charge_file.c_str(),
                    rec.topology_strand,
                    _protein_topology_strand[
                        rec.local_index
                    ]
                );
            }

            if(rec.residue !=
               _protein_residue_code[
                   rec.local_index
               ]) {

                throw oxDNAException(
                    "Protein residue mismatch for local index %d on line %d of '%s': file=%c topology=%c",
                    rec.local_index,
                    rec.line_no,
                    _protein_charge_file.c_str(),
                    rec.residue,
                    _protein_residue_code[
                        rec.local_index
                    ]
                );
            }
        }

        const int global_index =
            _protein_local_to_global[
                rec.local_index
            ];

        if(global_index < 0 ||
           global_index >= total_particles) {

            throw oxDNAException(
                "Internal protein index mapping failed for local index %d",
                rec.local_index
            );
        }

        const number reduced_charge =
            checked_reduced_protein_charge(
                _protein_charge_scale,
                rec.charge_e,
                _dna_effective_charge_e,
                rec.local_index
            );

        _protein_charge_reduced[
            global_index
        ] = reduced_charge;

        _protein_group[
            global_index
        ] = rec.group_id;

        seen[rec.local_index] = true;

        groups.insert(rec.group_id);

        nread++;

        net_charge_e +=
            (number) rec.charge_e;

        max_abs_charge_e =
            std::max(
                max_abs_charge_e,
                (number) std::fabs(
                    rec.charge_e
                )
            );
    }

    if(nread != npro) {
        throw oxDNAException(
            "Protein charge file '%s' contains %d entries, expected %d",
            _protein_charge_file.c_str(),
            nread,
            npro
        );
    }

    for(int i = 0; i < npro; i++) {
        if(!seen[i]) {
            throw oxDNAException(
                "Protein charge file '%s' is missing local index %d",
                _protein_charge_file.c_str(),
                i
            );
        }
    }

    auto ngroups_it =
        metadata.find("n_groups");

    if(format_version == 1 &&
       ngroups_it != metadata.end()) {

        int declared_groups = -1;

        std::stringstream gs(
            ngroups_it->second
        );

        std::string ngroups_extra;

        if(!(gs >> declared_groups) ||
           (gs >> ngroups_extra)) {
            throw oxDNAException(
                "Malformed integer metadata 'n_groups' in protein charge file '%s'",
                _protein_charge_file.c_str()
            );
        }

        if(declared_groups !=
           (int) groups.size()) {

            throw oxDNAException(
                "Version-1 protein charge file '%s' declares n_groups=%d but contains %d distinct groups",
                _protein_charge_file.c_str(),
                declared_groups,
                (int) groups.size()
            );
        }
    }

    _protein_charge_format_version =
        format_version;

    OX_LOG(
        Logger::LOG_INFO,
        "Loaded %d protein charges from %s "
        "(format=%s, model=%s, source=%s, "
        "groups=%d, unscaled net charge %.6g e, "
        "max |q| %.6g e)",
        nread,
        _protein_charge_file.c_str(),
        format_version == 1
            ? "v1"
            : "legacy",
        _protein_charge_model.c_str(),
        _protein_charge_source.c_str(),
        (int) groups.size(),
        net_charge_e,
        max_abs_charge_e
    );
}

void DNANMInteraction::_update_electrostatic_cutoffs() {

    _pro_pro_sqr_rcut =
        _pro_sqr_rcut;

    number required_sqr_cut =
        (number) 0.0;

    if(_protein_protein_electrostatics &&
       _protein_protein_dh_scale >
           (number) 0.0) {

        _pro_pro_sqr_rcut =
            std::max(
                _pro_pro_sqr_rcut,
                SQR(_debye_huckel_RC)
            );

        required_sqr_cut =
            std::max(
                required_sqr_cut,
                _pro_pro_sqr_rcut
            );
    }

    if(_protein_dna_electrostatics &&
       _protein_dna_dh_scale >
           (number) 0.0) {

        const number backbone_offset =
            (_grooving)
            ? sqrt(
                SQR(POS_MM_BACK1)
                +
                SQR(POS_MM_BACK2)
              )
            : std::fabs(
                (number) POS_BACK
              );

        const number pd_com_cut =
            backbone_offset
            +
            _debye_huckel_RC;

        _pro_dna_sqr_rcut =
            std::max(
                _pro_dna_sqr_rcut,
                SQR(pd_com_cut)
            );

        required_sqr_cut =
            std::max(
                required_sqr_cut,
                _pro_dna_sqr_rcut
            );
    }

    if(required_sqr_cut >
       (number) 0.0) {

        const number required_cut =
            sqrt(required_sqr_cut);

        if(required_cut > _rcut) {

            _rcut =
                required_cut;

            _sqr_rcut =
                SQR(_rcut);
        }
    }
}

number DNANMInteraction::_scaled_debye_huckel(
    const LR_vector &r,
    number charge_product,
    LR_vector &force,
    bool update_forces
) {

    force = LR_vector(0, 0, 0);

    if(charge_product ==
       (number) 0.0) {
        return (number) 0.0;
    }

    const number rmod =
        r.module();

    if(rmod <=
           std::numeric_limits<number>::epsilon()
       ||
       rmod >=
           _debye_huckel_RC) {

        return (number) 0.0;
    }

    number energy;

    if(rmod <
       _debye_huckel_RHIGH) {

        const number exponential =
            exp(
                _minus_kappa *
                rmod
            );

        energy =
            charge_product
            *
            exponential
            *
            (_debye_huckel_prefactor /
             rmod);

        if(update_forces) {

            force =
                (r / rmod)
                *
                (
                    charge_product
                    *
                    (
                        -_debye_huckel_prefactor
                        *
                        exponential
                    )
                    *
                    (
                        _minus_kappa /
                        rmod
                        -
                        (number) 1.0 /
                        SQR(rmod)
                    )
                );
        }
    }
    else {

        energy =
            charge_product
            *
            _debye_huckel_B
            *
            SQR(
                rmod -
                _debye_huckel_RC
            );

        if(update_forces) {

            force =
                -(r / rmod)
                *
                (
                    charge_product
                    *
                    (number) 2.0
                    *
                    _debye_huckel_B
                    *
                    (
                        rmod -
                        _debye_huckel_RC
                    )
                );
        }
    }

    return energy;
}

number DNANMInteraction::_protein_dna_debye_huckel(
    BaseParticle *p,
    BaseParticle *q,
    bool compute_r,
    bool update_forces
) {

    if(!_protein_dna_electrostatics) {
        return (number) 0.0;
    }

    BaseParticle *protein = nullptr;
    BaseParticle *nuc = nullptr;

    LR_vector rcenter =
        _computed_r;

    const int pid =
        get_id(p->btype);

    const int qid =
        get_id(q->btype);

    if(pid == 0 && qid == 1) {

        nuc = p;
        protein = q;
    }
    else if(pid == 1 &&
            qid == 0) {

        protein = p;
        nuc = q;

        rcenter =
            -rcenter;
    }
    else {
        return (number) 0.0;
    }

    if(protein->index < 0 ||
       protein->index >=
           (int) _protein_charge_reduced.size()) {

        throw oxDNAException(
            "DNANM protein-DNA electrostatics requested for protein particle %d outside charge array of size %d",
            protein->index,
            (int) _protein_charge_reduced.size()
        );
    }

    // DNA backbone carries negative effective charge.
    number charge_product =
        checked_number_product(
            -_protein_dna_dh_scale,
            _protein_charge_reduced[
                protein->index
            ],
            "Protein-DNA electrostatic charge factor"
        );

    if(_debye_huckel_half_charged_ends &&
       (
           nuc->n3 == P_VIRTUAL ||
           nuc->n5 == P_VIRTUAL
       )) {

        charge_product *=
            (number) 0.5;
    }

    const LR_vector rback =
        rcenter
        -
        nuc->int_centers[
            DNANucleotide::BACK
        ];

    LR_vector force_on_protein(
        0,
        0,
        0
    );

    const number energy =
        _scaled_debye_huckel(
            rback,
            charge_product,
            force_on_protein,
            update_forces
        );

    if(update_forces &&
       energy != (number) 0.0) {

        const LR_vector force_on_nuc =
            -force_on_protein;

        nuc->force +=
            force_on_nuc;

        protein->force +=
            force_on_protein;

        const LR_vector torque_nuc =
            nuc->int_centers[
                DNANucleotide::BACK
            ].cross(
                force_on_nuc
            );

        nuc->torque +=
            nuc->orientationT
            *
            torque_nuc;

        // Current upstream particle-aware stress API.
        // rcenter = protein COM - nucleotide COM.
        // force_on_protein = force acting on second particle.
        _update_stress_tensor(
            nuc,
            protein,
            rcenter,
            force_on_protein
        );
    }

    return energy;
}

number DNANMInteraction::_protein_protein_debye_huckel(
    BaseParticle *p,
    BaseParticle *q,
    bool compute_r,
    bool update_forces
) {

    if(!_protein_protein_electrostatics ||
       get_id(p->btype) != 1 ||
       get_id(q->btype) != 1) {

        return (number) 0.0;
    }

    // Direct term API and nonbonded lists must agree: bonded ANM springs
    // contribute through bonded mechanics, never screened PP DH.
    if(p->is_bonded(q)) return (number) 0.0;
    if(p->index < 0 ||
       q->index < 0 ||
       p->index >=
           (int) _protein_charge_reduced.size() ||
       q->index >=
           (int) _protein_charge_reduced.size()) {

        throw oxDNAException(
            "DNANM protein-protein electrostatics requested for particles %d/%d outside charge array of size %d",
            p->index,
            q->index,
            (int) _protein_charge_reduced.size()
        );
    }

    if(_exclude_intragroup_electrostatics &&
       _protein_group[p->index] ==
       _protein_group[q->index]) {

        return (number) 0.0;
    }

    const number scaled_p_charge =
        checked_number_product(
            _protein_protein_dh_scale,
            _protein_charge_reduced[p->index],
            "Protein-protein electrostatic scale/charge product"
        );

    const number charge_product =
        checked_number_product(
            scaled_p_charge,
            _protein_charge_reduced[q->index],
            "Protein-protein electrostatic charge product"
        );

    LR_vector force_on_q(
        0,
        0,
        0
    );

    const number energy =
        _scaled_debye_huckel(
            _computed_r,
            charge_product,
            force_on_q,
            update_forces
        );

    if(update_forces &&
       energy != (number) 0.0) {

        p->force -=
            force_on_q;

        q->force +=
            force_on_q;

        // Current upstream particle-aware stress API.
        _update_stress_tensor(
            p,
            q,
            _computed_r,
            force_on_q
        );
    }

    return energy;
}

number DNANMInteraction::pair_interaction_term(
    int name,
    BaseParticle *p,
    BaseParticle *q,
    bool compute_r,
    bool update_forces
) {

    const int interaction_type =
        get_id(p->btype)
        +
        get_id(q->btype);

    const bool bonded =
        p->is_bonded(q);

    BaseParticle *left = p;
    BaseParticle *right = q;

    // A supplied cache is oriented from the caller's p to q. Canonical PP
    // endpoints require the opposite vector when that order is reversed.
    // Restore the caller's cache on every return and on exceptions so repeated
    // term queries cannot alternately flip or silently invalidate its geometry.
    const bool reverse_cached_r = interaction_type == 2 &&
        left->index > right->index && !compute_r;
    struct CachedDisplacementGuard {
        LR_vector &target;
        LR_vector saved;
        bool restore;
        ~CachedDisplacementGuard() {
            if(restore) target = saved;
        }
    } cached_displacement_guard{
        _computed_r,
        reverse_cached_r ? _computed_r : LR_vector(0, 0, 0),
        reverse_cached_r
    };

    if(interaction_type == 2 &&
       left->index > right->index) {

        std::swap(left, right);
        if(reverse_cached_r) _computed_r = -_computed_r;
    }

    if(compute_r) {
        _computed_r =
            this->_box->min_image(
                left->pos,
                right->pos
            );
    }

    if(interaction_type == 0) {

        const bool bonded_term =
            name == BACKBONE
            ||
            name == BONDED_EXCLUDED_VOLUME
            ||
            name == STACKING;

        const bool nonbonded_term =
            name == NONBONDED_EXCLUDED_VOLUME
            ||
            name == HYDROGEN_BONDING
            ||
            name == CROSS_STACKING
            ||
            name == COAXIAL_STACKING
            ||
            name == DEBYE_HUCKEL;

        if(
            (bonded && bonded_term)
            ||
            (!bonded && nonbonded_term)
        ) {

            return BaseInteraction::
                pair_interaction_term(
                    name,
                    p,
                    q,
                    false,
                    update_forces
                );
        }

        return (number) 0.0;
    }

    if(interaction_type == 1) {

        if(!bonded &&
           (
               name == PRO_DNA_EXC_VOL
               ||
               (
                   name == PRO_DNA_DH
                   &&
                   _protein_dna_electrostatics
               )
           )) {

            return BaseInteraction::
                pair_interaction_term(
                    name,
                    p,
                    q,
                    false,
                    update_forces
                );
        }

        return (number) 0.0;
    }

    if(interaction_type == 2) {

        if(name == SPRING) {
            return bonded
                ? _protein_spring(
                    left,
                    right,
                    false,
                    update_forces
                  )
                : (number) 0.0;
        }

        if(name == PRO_EXC_VOL) {
            return _protein_exc_volume(
                left,
                right,
                false,
                update_forces
            );
        }

        if(name == PRO_PRO_DH) {

            if(bonded ||
               !_protein_protein_electrostatics) {

                return (number) 0.0;
            }

            return _protein_protein_debye_huckel(
                left,
                right,
                false,
                update_forces
            );
        }

        if(name == PRO_ANG_POT) {

            if(!_angular ||
               !bonded ||
               right->index -
               left->index != 1) {

                return (number) 0.0;
            }

            return _protein_ang_pot(
                left,
                right,
                false,
                update_forces
            );
        }
    }

    return (number) 0.0;
}

number DNANMInteraction::pair_interaction(BaseParticle *p, BaseParticle *q, bool compute_r, bool update_forces){
    int interaction_type = get_id(p->btype) + get_id(q->btype);

    if (interaction_type == 0 || interaction_type == 2){
        if(p->is_bonded(q)) return pair_interaction_bonded(p, q, compute_r, update_forces);
        else return pair_interaction_nonbonded(p, q, compute_r, update_forces);
    }
    if (interaction_type == 1) return this->pair_interaction_nonbonded(p, q, compute_r, update_forces);

    return 0.f;
}


number DNANMInteraction::pair_interaction_bonded(BaseParticle *p, BaseParticle *q, bool compute_r, bool update_forces) {
    if(compute_r)
        if (q != P_VIRTUAL && p != P_VIRTUAL)
            _computed_r = this->_box->min_image(p->pos, q->pos);

    int interaction_type = get_id(p->btype) + get_id(q->btype);
    if (interaction_type == 0){ // dna-dna
        if(!this->_check_bonded_neighbour(&p, &q, compute_r)) return (number) 0;
        number energy = _backbone(p,q,compute_r,update_forces);
        energy += _bonded_excluded_volume(p,q,compute_r,update_forces);
        energy += _stacking(p,q,compute_r,update_forces);
        return energy;
    } else if (interaction_type == 2){ // protein-protein
        if(_angular){
            if (!p->is_bonded(q)) return 0.f;
                // C03 diagnostic: adjacent angular protein bonds have a
                // canonical index order, independent of pair-list order.
                // The precomputed r may refer to the reversed order.
                if (p->index == q->index + 1) {
                    BaseParticle *first = q;
                    q = p;
                    p = first;
                    _computed_r = this->_box->min_image(p->pos, q->pos);
                }
            number energy = _protein_spring(p, q, compute_r, update_forces);
            energy += _protein_exc_volume(p, q, compute_r, update_forces);
            if (q->index - p->index == 1) energy += _protein_ang_pot(p, q, compute_r, update_forces);
            return energy;
        } else {
            if (!p->is_bonded(q)) return 0.f;
            number energy = _protein_spring(p, q, compute_r, update_forces);
            energy += _protein_exc_volume(p, q, compute_r, update_forces);
            return energy;
        }
    } else
        return 0.f;

    // Expect Topology to not contain Bonds b/t protein & DNA
}


number DNANMInteraction::pair_interaction_nonbonded(BaseParticle *p, BaseParticle *q, bool compute_r, bool update_forces) {
    if (compute_r)
        _computed_r = this->_box->min_image(p->pos, q->pos);

    number rnorm = _computed_r.norm();

    int interaction_type = get_id(p->btype) + get_id(q->btype);

    if (interaction_type == 0) { //DNA-DNA Interaction
        if (rnorm >= _sqr_rcut) return (number) 0.f;
        number energy = _nonbonded_excluded_volume(p, q, compute_r, update_forces);
        energy += _hydrogen_bonding(p, q, compute_r, update_forces);
        energy += _cross_stacking(p, q, compute_r, update_forces);
        energy += _coaxial_stacking(p, q, compute_r, update_forces);
        energy += _debye_huckel(p, q, compute_r, update_forces);
        return energy;
    } else if (interaction_type == 2) { // protein-protein
        if (rnorm >= (_protein_protein_electrostatics ? _pro_pro_sqr_rcut : _pro_sqr_rcut))
            return (number) 0.f;
        number energy = _protein_exc_volume(p, q, compute_r, update_forces);
        if(_protein_protein_electrostatics) {
            energy += _protein_protein_debye_huckel(p, q, false, update_forces);
        }
        return energy;
//        return 0.f;
    }else if(interaction_type == 1) { //protein-dna
        if (rnorm >= _pro_dna_sqr_rcut) return (number) 0.f;
        number energy = _protein_dna_exc_volume(p, q, compute_r, update_forces);
        if(_protein_dna_electrostatics) {
            energy += _protein_dna_debye_huckel(p, q, false, update_forces);
        }
        return energy;
    }

    return 0.f;
}


number DNANMInteraction::_protein_dna_exc_volume(BaseParticle *p, BaseParticle *q, bool compute_r, bool update_forces)
{
    BaseParticle *protein;
    BaseParticle *nuc;

    LR_vector force(0, 0, 0);
    LR_vector rcenter = _computed_r;
    int pid = get_id(p->btype);
    int qid = get_id(q->btype);

    if(pid == 0 && qid == 1)
    {
        //rcenter = -rcenter;
        protein = q;
        nuc = p;
    }
    else if (pid == 1 && qid == 0)
    {
        rcenter = -rcenter;
        protein = p;
        nuc = q;
    }
    else
        return 0.f;

    LR_vector r_to_back = rcenter  - nuc->int_centers[DNANucleotide::BACK];
    LR_vector r_to_base = rcenter  - nuc->int_centers[DNANucleotide::BASE];

    LR_vector torquenuc(0,0,0);
    auto energy = (number) 0.f;


    if(r_to_back.norm() < _pro_backbone_sqr_rcut) {
        energy = _protein_dna_repulsive_lj(r_to_back, force, update_forces, _pro_backbone_sigma, _pro_backbone_b, _pro_backbone_rstar,_pro_backbone_rcut,_pro_backbone_stiffness);
        //printf("back-pro %d %d %f\n",p->index,q->index,energy);
        if (update_forces) {
            torquenuc = nuc->int_centers[DNANucleotide::BACK].cross(-force);
            nuc->torque += nuc->orientationT * torquenuc;
            nuc->force -= force;
            protein->force += force;
            _update_stress_tensor(nuc, protein, rcenter, force);
            if(_angular){
                r_to_back.normalize();
                protein->torque += p->orientationT*(-r_to_back*_pro_sigma*0.5f).cross(force); // dr Point of contact on protein particle relative to COM of protein particle
            }
        }
    }

    if(r_to_base.norm() < _pro_base_sqr_rcut){
//        printf("pro %d dna %d\n", protein->index, nuc->index);
        energy += _protein_dna_repulsive_lj(r_to_base, force, update_forces, _pro_base_sigma, _pro_base_b, _pro_base_rstar, _pro_base_rcut, _pro_base_stiffness);
        if(update_forces) {
            torquenuc = nuc->int_centers[DNANucleotide::BASE].cross(-force);
            nuc->torque += nuc->orientationT * torquenuc;
            nuc->force -= force;
            protein->force += force;
            _update_stress_tensor(nuc, protein, rcenter, force);
            if(_angular){
                r_to_base.normalize();
                protein->torque += p->orientationT*(-r_to_base*_pro_sigma*0.5f).cross(force); // dr Point of contact on protein particle relative to COM of protein particle
            }
        }
    }

    return energy;
}


number DNANMInteraction::_protein_dna_repulsive_lj(const LR_vector &r, LR_vector &force, bool update_forces, number &sigma, number &b, number &rstar, number &rcut, number &stiffness) {
    // this is a bit faster than calling r.norm()
    number rnorm = SQR(r.x) + SQR(r.y) + SQR(r.z);
    number energy = (number) 0;

    if(rnorm < SQR(rcut)) {
        if(rnorm > SQR(rstar)) {
            number rmod = sqrt(rnorm);
            number rrc = rmod - rcut;
            energy = stiffness * b * SQR(SQR(rrc));
            if(update_forces) force = -r * (4 * stiffness * b * CUB(rrc) / rmod);
        }
        else {
            number tmp = SQR(sigma) / rnorm;
            number lj_part = tmp * tmp * tmp;
            energy = 4 * stiffness * (SQR(lj_part) - lj_part);
            if(update_forces) force = -r * (24 * stiffness * (lj_part - 2*SQR(lj_part)) / rnorm);
        }
    }

    if(update_forces && energy == (number) 0) force.x = force.y = force.z = (number) 0;

    return energy;
}


void DNANMInteraction::init() {
    DNA2Interaction::init();
    ndna=0, npro=0, ndnas =0;
    //let's try this
    _pro_backbone_sigma = 0.57f;
    _pro_backbone_rstar= 0.569f;
    _pro_backbone_b = 178699253.5f;
    _pro_backbone_rcut = 0.572934f;
    _pro_backbone_stiffness = 1.0f;
    _pro_backbone_sqr_rcut = 0.3283f;
    //Base-Protein Excluded Volume Parameters
    _pro_base_sigma = 0.36f;
    _pro_base_rstar= 0.359f;
    _pro_base_b = 296866090.f;
    _pro_base_rcut = 0.362897f;
    _pro_base_stiffness = 1.0f;
    _pro_base_sqr_rcut = 0.1317f;
    //Protein-Protein Excluded Volume Parameters
    _pro_sigma = 0.35f;
    _pro_rstar = 0.349f;
    _pro_b = 306484596.421f;
    _pro_rcut = 0.352894;
    _pro_sqr_rcut = 0.12454f; //_rc ^2

    _pro_dna_sqr_rcut = 3.0625f; //placeholder test value 1.75^2
    _pro_pro_sqr_rcut = _pro_sqr_rcut;
    _update_electrostatic_cutoffs();
}

//Functions almost Identical to those in ANMInteraction
number DNANMInteraction::_protein_repulsive_lj(const LR_vector &r, LR_vector &force, bool update_forces) {
    // this is a bit faster than calling r.norm()
    //changed to a quartic form
    number rnorm = SQR(r.x) + SQR(r.y) + SQR(r.z);
    number energy = (number) 0;
    if(rnorm < SQR(_pro_rcut)) {
        if(rnorm > SQR(_pro_rstar)) {
            number rmod = sqrt(rnorm);
            number rrc = rmod - _pro_rcut;
            energy = EXCL_EPS * _pro_b * SQR(SQR(rrc));
            if(update_forces) force = -r * (4 * EXCL_EPS * _pro_b * CUB(rrc)/ rmod);
        }
        else {
            number tmp = SQR(_pro_sigma) / rnorm;
            number lj_part = tmp * tmp * tmp;
            energy = 4 * EXCL_EPS * (SQR(lj_part) - lj_part);
            if(update_forces) force = -r* (24 * EXCL_EPS * (lj_part - 2*SQR(lj_part))/rnorm);
        }
    }

    if(update_forces && energy == (number) 0) force.x = force.y = force.z = (number) 0;

    return energy;
}


number DNANMInteraction::_protein_exc_volume(BaseParticle *p, BaseParticle *q, bool compute_r, bool update_forces) {
    LR_vector force(0,0,0);
    number energy =  _protein_repulsive_lj(_computed_r, force, update_forces);

    if(update_forces)
    {
        p->force -= force;
        q->force += force;
        _update_stress_tensor(p, q, _computed_r, force);
        if(_angular){
            p->torque += p->orientationT*(_computed_r*0.5).cross(-force);
            q->torque += q->orientationT*(-_computed_r*0.5).cross(force);
        }
    }

    return energy;
}


number DNANMInteraction::_protein_spring(BaseParticle *p, BaseParticle *q, bool compute_r, bool update_forces) {

    std::pair <int,int> keys (std::min(p->index, q->index), std::max(p->index, q->index));

    //Harmonic Spring Potential
    number _k = _potential[keys].second; //stiffness of the spring

    number rinsta = _computed_r.module(); // distance b/t p and q
    number disp = rinsta - _rknot[keys]; // current distance - eqdistance
    number energy = 0.5 * _k * SQR(disp);

    if (update_forces) {
        LR_vector force(_computed_r);
        force *= (-1.0f * _k ) * disp/rinsta;

        p->force -= force;
        q->force += force;
        _update_stress_tensor(p, q, _computed_r, force);
    }

    return energy;
}


number DNANMInteraction::_protein_ang_pot(BaseParticle *p, BaseParticle *q, bool compute_r, bool update_forces) {
    // Get Angular Parameters
    std::vector<double> &ang_params = _ang_vals[p->index];
    double &a0 = ang_params[0];
    double &b0 = ang_params[1];
    double &c0 = ang_params[2];
    double &d0 = ang_params[3];

    if (_parameter_kbkt) { //Uses array with per particle kb and kt
        _k_bend = this->_ang_stiff[p->index].first;
        _k_tor = this->_ang_stiff[p->index].second;
    } // If False, the global kb and kt will be used


    LR_vector rij_unit = _computed_r;
    rij_unit.normalize();
    LR_vector rji_unit = rij_unit * -1.f;

    LR_vector &a1 = p->orientationT.v1;
    LR_vector &b1 = q->orientationT.v1;
    LR_vector &a3 = p->orientationT.v3;
    LR_vector &b3 = q->orientationT.v3;


    double o1 = rij_unit * a1 - a0;
    double o2 = rji_unit * b1 - b0;
    double o3 = a1 * b1 - c0;
    double o4 = a3 * b3 - d0;

    //Torsion and Bending
    number energy = _k_bend * 0.5f * (SQR(o1) + SQR(o2)) + _k_tor * 0.5f * (SQR(o3) + SQR(o4));

    if (update_forces) {

        LR_vector force = -(rji_unit.cross(rij_unit.cross(a1)) * _k_bend * o1 -
                                    rji_unit.cross(rij_unit.cross(b1)) * _k_bend * o2) / _computed_r.module();

        p->force -= force;
        q->force += force;
        // Angular bending contributes to configurational virial.
        // Record one pair contribution, half-shared between endpoints.
        _update_stress_tensor(p, q, _computed_r, force);

        LR_vector ta = rij_unit.cross(a1) * o1 * _k_bend;
        LR_vector tb = rji_unit.cross(b1) * o2 * _k_bend;

        p->torque += p->orientationT * ta;
        q->torque += q->orientationT * tb;

        LR_vector torsion = a1.cross(b1) * o3 * _k_tor + a3.cross(b3) * o4 * _k_tor;

        p->torque += p->orientationT * -torsion;
        q->torque += q->orientationT * torsion;

        //For Debugging, very helpful for CUDA comparisons
        /*LR_vector<number> TA = p->orientationT * (ta - torsion);
        LR_vector<number> TB = q->orientationT * (tb + torsion);

        if (p->index ==104){
            printf("p2 Angular F %.7f %.7f %.7f TA %.7f %.7f %.7f\n", -force.x, -force.y, -force.z, TA.x, TA.y, TA.z);
        } else if (q->index ==104){
            printf("q2 Angular F %.7f %.7f %.7f TB %.7f %.7f %.7f\n", force.x, force.y, force.z, TB.x, TB.y, TB.z);
        }*/

    }

    return energy;
}


void DNANMInteraction::load_massfile(std::string &filename) {
    // Hybrid integration, thermostats and kinetic observables assume unit mass.
    // Retain legacy unit-mass tables without advertising unsupported dynamics.
    std::ifstream mass_stream(filename);
    if(!mass_stream.is_open())
        throw oxDNAException("Could Not Load Mass File, Aborting");

    std::map<int, number> parsed_masses;
    int expected_types = -1;
    int line_no = 0;
    std::string row;
    while(std::getline(mass_stream, row)) {
        ++line_no;
        const std::string trimmed = trim_copy(row.substr(0, row.find('#')));
        if(trimmed.empty()) continue;
        std::istringstream fields(trimmed);
        std::string extra;
        if(expected_types < 0) {
            if(!(fields >> expected_types) || expected_types <= 0 || (fields >> extra))
                throw oxDNAException("Invalid massfile header on line %d: expected a positive type count", line_no);
            continue;
        }
        int type = -1;
        std::string mass_text;
        if(!(fields >> type >> mass_text) || type < 0 || (fields >> extra))
            throw oxDNAException("Malformed massfile record on line %d", line_no);
        const double mass = parse_finite_double_text(
            mass_text, std::string("Protein mass on line ") + std::to_string(line_no));
        if(mass != 1.0)
            throw oxDNAException("DNANM/DNACT supports unit protein masses only; type %d on massfile line %d is not 1", type, line_no);
        if(!parsed_masses.emplace(type, (number) 1.0).second)
            throw oxDNAException("Duplicate massfile type %d on line %d", type, line_no);
    }
    if(mass_stream.bad())
        throw oxDNAException("I/O error reading massfile");
    if(expected_types <= 0 || parsed_masses.size() != static_cast<std::size_t>(expected_types))
        throw oxDNAException("Massfile type count does not match its header");
    // Publish only a complete table; read_topology rejects missing used types.
    masses.swap(parsed_masses);
}

int DNANMInteraction::get_id(int btype){
    // takes btype return whether dna or protein
    // 0 is dna
    // >=5 is protein
    return (btype <= 4) ? 0: 1;
};

DNANMInteraction::~DNANMInteraction() = default;




