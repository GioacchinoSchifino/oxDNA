/*
 * DNANMInteraction.cpp
 * Original ANM: Jonah, Subhajit; shared ANM architecture: Erik Poppleton.
 * Optional CPU screened protein-DNA / protein-protein electrostatics:
 * Gioacchino Schifino (2026), ported to Erik's ANM refactor.
 *
 * This class intentionally delegates ALL unmodified forces to
 * ANMHybridInteraction<DNA2Interaction>. ES=OFF uses the parent's exact path.
 */

#include "DNANMInteraction.h"
#include "../Particles/DNANucleotide.h"
#include "../Utilities/oxDNAException.h"
#include "../Utilities/Logger.h"

#include <algorithm>
#include <cerrno>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <map>
#include <set>
#include <sstream>

namespace {
std::string trim_es(const std::string &s) {
    const auto b = s.find_first_not_of(" \t\r\n");
    if(b == std::string::npos) return "";
    const auto e = s.find_last_not_of(" \t\r\n");
    return s.substr(b, e - b + 1);
}

// Lexical validation stays local: do not add a new general parser to oxDNA.
// Reject NaN, Inf, hex literals and trailing garbage even under -ffast-math.
double checked_es_number(const std::string &text, const std::string &context) {
    if(text.empty()) throw oxDNAException("Empty numeric value in %s", context.c_str());
    bool digit = false;
    for(char c : text) {
        if(std::isdigit(static_cast<unsigned char>(c))) digit = true;
        else if(c != '.' && c != '-' && c != '+' && c != 'e' && c != 'E')
            throw oxDNAException("Invalid numeric value '%s' in %s", text.c_str(), context.c_str());
    }
    if(!digit) throw oxDNAException("Invalid numeric value '%s' in %s", text.c_str(), context.c_str());
    errno = 0;
    char *end = nullptr;
    const double v = std::strtod(text.c_str(), &end);
    if(errno == ERANGE || end == text.c_str() || *end != '\0' ||
       !std::isfinite(v))
        throw oxDNAException("Nonfinite/out-of-range number '%s' in %s", text.c_str(), context.c_str());
    return v;
}

number checked_es_scalar(input_file &inp, const char *key, number fallback, bool positive) {
    std::string raw;
    if(getInputString(&inp, key, raw, 0) != KEY_FOUND) return fallback;
    const double v = checked_es_number(trim_es(raw), key);
    if((positive && !(v > 0.0)) || (!positive && v < 0.0) ||
       v > static_cast<double>(std::numeric_limits<number>::max()) ||
       (v != 0.0 && static_cast<number>(v) == (number) 0.0))
        throw oxDNAException("Invalid range for %s", key);
    return static_cast<number>(v);
}

number checked_es_product(long double value, const char *context) {
    const long double mx = static_cast<long double>(std::numeric_limits<number>::max());
    if(!std::isfinite(value) || std::fabs(value) > mx)
        throw oxDNAException("Electrostatic charge product overflow in %s", context);
    const number out = static_cast<number>(value);
    if(value != 0.0L && out == (number) 0.0)
        throw oxDNAException("Electrostatic charge product underflow in %s", context);
    return out;
}
}

DNANMInteraction::DNANMInteraction(bool torsion) :
    ANMHybridInteraction<DNA2Interaction>(torsion, "DNA") {
    ADD_INTERACTION_TO_MAP(PRO_DNA_DH, _term_protein_dna_dh);
    ADD_INTERACTION_TO_MAP(PRO_PRO_DH, _term_protein_protein_dh);
}

DNANMInteraction::~DNANMInteraction() = default;

BaseParticle *DNANMInteraction::_new_nucleotide(int index, char kind) {
    return new DNANucleotide(this->_grooving);
}

void DNANMInteraction::get_settings(input_file &inp) {
    ANMHybridInteraction<DNA2Interaction>::get_settings(inp);

    // A single switch, per Erik's review; older split switches must not be
    // silently accepted, since that would run the wrong Hamiltonian.
    bool old_pd = false, old_pp = false;
    if(getInputBool(&inp, "protein_dna_electrostatics", &old_pd, 0) == KEY_FOUND ||
       getInputBool(&inp, "protein_protein_electrostatics", &old_pp, 0) == KEY_FOUND)
        throw oxDNAException("Use protein_electrostatics (single switch); the old PD/PP switches are unsupported");

    getInputBool(&inp, "protein_electrostatics", &_protein_electrostatics, 0);
    if(!_protein_electrostatics) return;

    if(getInputString(&inp, "protein_charge_file", _protein_charge_file, 0) != KEY_FOUND ||
       _protein_charge_file.empty())
        throw oxDNAException("protein_charge_file is required when protein_electrostatics=true");

    getInputBool(&inp, "exclude_intragroup_electrostatics", &_exclude_intragroup_electrostatics, 0);
    _protein_charge_scale = checked_es_scalar(inp, "protein_charge_scale", _protein_charge_scale, false);
    _protein_dna_dh_scale = checked_es_scalar(inp, "protein_dna_dh_scale", _protein_dna_dh_scale, false);
    _protein_protein_dh_scale = checked_es_scalar(inp, "protein_protein_dh_scale", _protein_protein_dh_scale, false);
    _dna_effective_charge_e = checked_es_scalar(inp, "dna_effective_charge_e", _dna_effective_charge_e, true);

    if(!std::isfinite(_debye_huckel_prefactor) || _debye_huckel_prefactor <= 0)
        throw oxDNAException("Protein electrostatics require a finite positive dh_strength");
    if(!std::isfinite(_salt_concentration) || _salt_concentration <= 0)
        throw oxDNAException("Protein electrostatics require a finite positive salt_concentration");
}

void DNANMInteraction::init() {
    ANMHybridInteraction<DNA2Interaction>::init();
    if(!_protein_electrostatics) return;

    // A protein bead acts at its COM, a nucleotide at its backbone site.
    // Expand the GLOBAL neighbour-list cutoff, not the ANM excluded-volume
    // cutoffs, which must remain unchanged.
    const number offset = _grooving ?
        std::sqrt(SQR(POS_MM_BACK1) + SQR(POS_MM_BACK2)) :
        std::fabs((number) POS_BACK);
    const number needed = std::max(_debye_huckel_RC, _debye_huckel_RC + offset);
    if(needed > _rcut) {
        _rcut = needed;
        _sqr_rcut = SQR(_rcut);
    }
}

void DNANMInteraction::read_topology(int *N_strands, std::vector<BaseParticle *> &particles) {
    ANMHybridInteraction<DNA2Interaction>::read_topology(N_strands, particles);
    if(_protein_electrostatics) _load_protein_charges(particles.size());
}

void DNANMInteraction::_load_protein_charges(size_t n_particles) {
    const auto &top = _anm_topology; // parsed once by Erik's shared ANMTopology
    if(top.kinds.size() != n_particles || top.N_amino_acids <= 0)
        throw oxDNAException("Protein electrostatics require a valid ANM protein topology");

    std::vector<int> global_ids;
    std::vector<int> expected_strands(n_particles, 0);
    std::vector<char> expected_codes(n_particles, '\0');

    if(top.new_style) {
        size_t i = 0;
        int protein_ordinal = 0;
        for(const auto &strand : top.strands) {
            const int label = strand.peptide ? -(++protein_ordinal) : 0;
            for(char code : strand.sequence) {
                if(i >= n_particles) throw oxDNAException("Invalid ANM strand lengths");
                if(strand.peptide) {
                    if(top.kinds[i] != 'P') throw oxDNAException("ANM protein kind mismatch");
                    global_ids.push_back(static_cast<int>(i));
                    expected_strands[i] = label;
                    expected_codes[i] = static_cast<char>(std::toupper(static_cast<unsigned char>(code)));
                }
                ++i;
            }
        }
        if(i != n_particles) throw oxDNAException("Incomplete ANM strand information");
    }
    else {
        if(top.old_particles.size() != n_particles) throw oxDNAException("Invalid old ANM topology");
        for(size_t i = 0; i < n_particles; ++i) {
            if(top.kinds[i] == 'P') {
                global_ids.push_back(static_cast<int>(i));
                expected_strands[i] = top.old_particles[i].strand;
                expected_codes[i] = static_cast<char>(std::toupper(
                    static_cast<unsigned char>(top.old_particles[i].code)));
            }
        }
    }
    if(global_ids.size() != static_cast<size_t>(top.N_amino_acids))
        throw oxDNAException("ANM protein count mismatch in charge mapping");

    std::ifstream in(_protein_charge_file);
    if(!in) throw oxDNAException("Cannot open protein charge file '%s'", _protein_charge_file.c_str());

    _protein_charge_reduced.assign(n_particles, (number)0);
    _protein_group.assign(n_particles, -1);
    std::vector<bool> seen(global_ids.size(), false);
    std::map<std::string, std::string> meta;
    std::set<int> groups;
    bool header = false, saw_data = false;
    size_t entries = 0;
    int line_no = 0;
    std::string line;
    while(std::getline(in, line)) {
        ++line_no;
        line = trim_es(line);
        if(line.empty()) continue;
        if(line[0] == '#') {
            const std::string comment = trim_es(line.substr(1));
            if(comment.find("DNANM_ES_CHARGES") == 0) {
                if(header || saw_data || comment != "DNANM_ES_CHARGES 1")
                    throw oxDNAException("Invalid/duplicate charge format header on line %d", line_no);
                header = true;
            }
            else {
                const auto eq = comment.find('=');
                if(eq != std::string::npos) {
                    const std::string key = trim_es(comment.substr(0, eq));
                    const std::string val = trim_es(comment.substr(eq + 1));
                    if(!meta.emplace(key, val).second)
                        throw oxDNAException("Duplicate charge metadata '%s'", key.c_str());
                }
            }
            continue;
        }
        if(!header) throw oxDNAException("Missing DNANM_ES_CHARGES 1 header");
        saw_data = true;
        const auto hash = line.find('#');
        if(hash != std::string::npos) line = trim_es(line.substr(0, hash));
        if(line.empty()) continue;
        std::istringstream ss(line);
        int local, strand, group;
        std::string residue, rawcharge, trailing;
        if(!(ss >> local >> strand >> residue >> rawcharge >> group) || (ss >> trailing) ||
           residue.size() != 1 || local < 0 ||
           static_cast<size_t>(local) >= global_ids.size() || group < 0)
            throw oxDNAException("Malformed charge row on line %d", line_no);
        if(seen[local]) throw oxDNAException("Duplicate protein local index %d", local);
        const int global = global_ids[local];
        const char aa = static_cast<char>(std::toupper(static_cast<unsigned char>(residue[0])));
        if(strand != expected_strands[global] || aa != expected_codes[global])
            throw oxDNAException("Charge file/topology mismatch at protein local index %d", local);

        const double q = checked_es_number(rawcharge, "protein_charge_file");
        const long double reduced = (static_cast<long double>(q) *
            static_cast<long double>(_protein_charge_scale)) /
            static_cast<long double>(_dna_effective_charge_e);
        _protein_charge_reduced[global] = checked_es_product(reduced, "protein charge");
        _protein_group[global] = group;
        groups.insert(group);
        seen[local] = true;
        ++entries;
    }
    if(in.bad() || !header || entries != global_ids.size())
        throw oxDNAException("Incomplete, unreadable or invalid protein charge file");
    auto require = [&](const char *key, const std::string &value) {
        auto it = meta.find(key);
        if(it == meta.end() || it->second != value)
            throw oxDNAException("Protein charge metadata '%s' must be '%s'", key, value.c_str());
    };
    require("index_space", "protein_local");
    require("index_base", "0");
    require("n_protein_beads", std::to_string(global_ids.size()));
    if(meta.count("n_groups"))
        require("n_groups", std::to_string(groups.size()));

    // Validate the worst-case pair factor before running any forces.
    long double maxq = 0;
    for(number q : _protein_charge_reduced) maxq = std::max(maxq, std::fabs(static_cast<long double>(q)));
    checked_es_product(maxq * static_cast<long double>(_protein_dna_dh_scale), "PD scale");
    checked_es_product(maxq * maxq * static_cast<long double>(_protein_protein_dh_scale), "PP scale");
    OX_LOG(Logger::LOG_INFO, "Loaded %d protein charges with %d groups (ANM ES, CPU)",
           (int) entries, (int) groups.size());
}

number DNANMInteraction::_scaled_dh(const LR_vector &r, number charge_product,
                                    LR_vector &force, bool update_forces) {
    force = LR_vector(0, 0, 0);
    if(charge_product == (number) 0) return (number) 0;
    const number dist = r.module();
    if(dist >= _debye_huckel_RC) return (number) 0;
    if(dist <= std::numeric_limits<number>::epsilon())
        throw oxDNAException("Overlapping charged protein interaction sites");
    if(dist < _debye_huckel_RHIGH) {
        const number expo = std::exp(_minus_kappa * dist);
        const number energy = charge_product * expo * _debye_huckel_prefactor / dist;
        if(update_forces)
            force = (r / dist) *
                (-charge_product * _debye_huckel_prefactor * expo *
                 (_minus_kappa / dist - (number) 1 / SQR(dist)));
        return energy;
    }
    const number delta = dist - _debye_huckel_RC;
    const number energy = charge_product * _debye_huckel_B * SQR(delta);
    if(update_forces)
        force = -(r / dist) * (charge_product * (number) 2 * _debye_huckel_B * delta);
    return energy;
}

number DNANMInteraction::_term_protein_dna_dh(BaseParticle *p, BaseParticle *q,
                                                bool compute_r, bool update_forces) {
    if(!_protein_electrostatics || _is_protein(p) == _is_protein(q) || p->is_bonded(q))
        return (number) 0;
    if(compute_r) this->_computed_r = this->_box->min_image(p->pos, q->pos);
    BaseParticle *protein = _is_protein(p) ? p : q;
    BaseParticle *nuc = _is_protein(p) ? q : p;
    if(_protein_charge_reduced.size() <= static_cast<size_t>(protein->index))
        throw oxDNAException("Protein charge index is out of range");
    const LR_vector rcenter = _is_protein(p) ? -this->_computed_r : this->_computed_r;
    const LR_vector rback = rcenter - nuc->int_centers[DNANucleotide::BACK];
    number factor = checked_es_product(-static_cast<long double>(_protein_dna_dh_scale) *
                            _protein_charge_reduced[protein->index], "PD pair");
    if(_debye_huckel_half_charged_ends &&
       (nuc->n3 == P_VIRTUAL || nuc->n5 == P_VIRTUAL)) factor *= (number) 0.5;
    LR_vector fpro(0, 0, 0);
    const number energy = _scaled_dh(rback, factor, fpro, update_forces);
    if(update_forces && factor != (number) 0) {
        const LR_vector fnuc = -fpro;
        nuc->force += fnuc;
        protein->force += fpro;
        nuc->torque += nuc->orientationT *
            nuc->int_centers[DNANucleotide::BACK].cross(fnuc);
        this->_update_stress_tensor(nuc, protein, rcenter, fpro);
    }
    return energy;
}

number DNANMInteraction::_term_protein_protein_dh(BaseParticle *p, BaseParticle *q,
                                                    bool compute_r, bool update_forces) {
    if(!_protein_electrostatics || !_is_protein(p) || !_is_protein(q) || p->is_bonded(q))
        return (number) 0;
    if(_exclude_intragroup_electrostatics && _protein_group[p->index] == _protein_group[q->index])
        return (number) 0;
    if(compute_r) this->_computed_r = this->_box->min_image(p->pos, q->pos);
    const number factor = checked_es_product(
        static_cast<long double>(_protein_protein_dh_scale) *
        _protein_charge_reduced[p->index] * _protein_charge_reduced[q->index], "PP pair");
    LR_vector fq(0, 0, 0);
    const number energy = _scaled_dh(this->_computed_r, factor, fq, update_forces);
    if(update_forces && factor != (number) 0) {
        p->force -= fq;
        q->force += fq;
        this->_update_stress_tensor(p, q, this->_computed_r, fq);
    }
    return energy;
}

number DNANMInteraction::pair_interaction_nonbonded(BaseParticle *p, BaseParticle *q,
                                                     bool compute_r, bool update_forces) {
    // ES-OFF delegates exactly to Erik's implementation.
    number energy = ANMHybridInteraction<DNA2Interaction>::pair_interaction_nonbonded(
        p, q, compute_r, update_forces);
    if(!_protein_electrostatics) return energy;
    // When parent delegates a DNA-DNA pair, it still controls its normal force path.
    if(_is_protein(p) && _is_protein(q))
        energy += _term_protein_protein_dh(p, q, false, update_forces);
    else if(_is_protein(p) != _is_protein(q))
        energy += _term_protein_dna_dh(p, q, false, update_forces);
    return energy;
}
