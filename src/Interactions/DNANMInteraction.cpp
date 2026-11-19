/*
 * DNANMInteraction.cpp
 *
 * DNA/ANM interaction: Jonah, Subhajit; refactored by Erik Nov 2026.
 * Optional CPU-only screened protein electrostatics: Gioacchino Schifino.
 *
 * This extension adds terms to the shared ANMHybridInteraction dispatcher;
 * it does not duplicate the ANM or nucleic-acid force fields.
 */

#include "DNANMInteraction.h"

#include "../Particles/DNANucleotide.h"
#include "../Utilities/Logger.h"
#include "../Utilities/oxDNAException.h"
#include "../Utilities/parse_input/strict_es_numeric.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <fstream>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace {

// Generic strict numeric parsing is kept in Utilities/parse_input.
using oxdna_es_numeric::es_trim;
using oxdna_es_numeric::es_finite_double;
using oxdna_es_numeric::es_int;
using oxdna_es_numeric::es_checked_number;
using oxdna_es_numeric::es_optional_number;

struct ChargeEntry {
	int local = -1;
	int strand = 0;
	int group = -1;
	char residue = '\0';
	double charge = 0.;
};

} // namespace

DNANMInteraction::DNANMInteraction(bool torsion) :
				ANMHybridInteraction<DNA2Interaction>(torsion, "DNA") {
}

DNANMInteraction::~DNANMInteraction() = default;

BaseParticle *DNANMInteraction::_new_nucleotide(int index, char kind) {
	return new DNANucleotide(this->_grooving);
}

void DNANMInteraction::get_settings(input_file &inp) {
	ANMHybridInteraction<DNA2Interaction>::get_settings(inp);

	// The public PR #229 had two switches; intentionally reject them instead
	// of silently running old input files with a different Hamiltonian.
	std::string deprecated;
	if(getInputString(&inp, "protein_dna_electrostatics", deprecated, 0) == KEY_FOUND ||
			getInputString(&inp, "protein_protein_electrostatics", deprecated, 0) == KEY_FOUND) {
		throw oxDNAException("Use protein_electrostatics (single switch); old protein_dna_electrostatics/protein_protein_electrostatics keys are no longer supported");
	}

	getInputBool(&inp, "protein_electrostatics", &_protein_electrostatics, 0);
	if(!_protein_electrostatics) return;

	std::string backend;
	if(getInputString(&inp, "backend", backend, 0) == KEY_FOUND) {
		std::transform(backend.begin(), backend.end(), backend.begin(),
				[](unsigned char c) { return static_cast<char>(std::toupper(c)); });
		if(es_trim(backend) == "CUDA") {
			throw oxDNAException("Screened ANM protein electrostatics are CPU-only; CUDA ES-ON forces are not implemented");
		}
	}

	getInputBool(&inp, "exclude_intragroup_electrostatics", &_exclude_intragroup_electrostatics, 0);
	es_optional_number(inp, "protein_charge_scale", _protein_charge_scale);
	es_optional_number(inp, "protein_dna_dh_scale", _protein_dna_dh_scale);
	es_optional_number(inp, "protein_protein_dh_scale", _protein_protein_dh_scale);
	es_optional_number(inp, "dna_effective_charge_e", _dna_effective_charge_e);

	if(_protein_charge_scale < 0 || _protein_dna_dh_scale < 0 ||
			_protein_protein_dh_scale < 0 || _dna_effective_charge_e <= 0) {
		throw oxDNAException("Protein electrostatics scales must be non-negative and dna_effective_charge_e strictly positive");
	}
	if(getInputString(&inp, "protein_charge_file", _protein_charge_file, 1) != KEY_FOUND ||
			_protein_charge_file.empty()) {
		throw oxDNAException("protein_charge_file is required with protein_electrostatics=true");
	}

	// These IDs deliberately follow the shared ANM interaction terms (8..11).
	ADD_INTERACTION_TO_MAP(PRO_DNA_SCREENED, _protein_dna_screened);
	ADD_INTERACTION_TO_MAP(PRO_PRO_SCREENED, _protein_protein_screened);
	OX_LOG(Logger::LOG_INFO, "CPU screened protein electrostatics enabled (protein-DNA and protein-protein)");
}

void DNANMInteraction::init() {
	ANMHybridInteraction<DNA2Interaction>::init();
	if(!_protein_electrostatics) return;

	// ANMModel's excluded-volume cutoffs are shorter than Debye-Huckel.
	// A nucleotide's backbone charge is displaced from its centre.
	const number backbone_offset = this->_grooving ?
			std::sqrt(SQR(POS_MM_BACK1) + SQR(POS_MM_BACK2)) :
			std::fabs((number) POS_BACK);

	const number pp_cut = (_protein_protein_dh_scale > 0) ? _debye_huckel_RC : 0;
	const number pd_cut = (_protein_dna_dh_scale > 0) ? _debye_huckel_RC + backbone_offset : 0;
	this->_rcut = std::max(this->_rcut, std::max(pp_cut, pd_cut));
	this->_sqr_rcut = SQR(this->_rcut);
}

void DNANMInteraction::read_topology(int *N_strands, std::vector<BaseParticle *> &particles) {
	ANMHybridInteraction<DNA2Interaction>::read_topology(N_strands, particles);
	if(_protein_electrostatics) _read_protein_charges();
}

void DNANMInteraction::_read_protein_charges() {
	const auto &top = this->_anm_topology;
	if(top.N_amino_acids <= 0) {
		throw oxDNAException("Protein electrostatics requires at least one protein bead");
	}

	std::vector<std::pair<int, char>> expected;
	expected.reserve(top.N_amino_acids);
	if(top.new_style) {
		int peptide_ordinal = 0;
		for(const auto &strand : top.strands) {
			if(!strand.peptide) continue;
			// Version-1 charge files use negative peptide ordinals for new
			// topology strands: first peptide=-1, second=-2, etc.
			const int label = -(++peptide_ordinal);
			for(char residue : strand.sequence) {
				expected.emplace_back(label, static_cast<char>(std::toupper((unsigned char) residue)));
			}
		}
	}
	else {
		for(const auto &record : top.old_particles) {
			if(record.peptide) {
				expected.emplace_back(record.strand,
						static_cast<char>(std::toupper((unsigned char) record.code)));
			}
		}
	}

	std::vector<int> global_indices;
	for(size_t i = 0; i < top.kinds.size(); ++i) {
		if(top.kinds[i] == 'P') global_indices.push_back(static_cast<int>(i));
	}
	if(expected.size() != global_indices.size() ||
			expected.size() != static_cast<size_t>(top.N_amino_acids)) {
		throw oxDNAException("Inconsistent ANM topology-to-charge mapping");
	}

	std::ifstream in(_protein_charge_file);
	if(!in) throw oxDNAException("Cannot open protein charge file '%s'", _protein_charge_file.c_str());

	std::vector<ChargeEntry> entries;
	std::map<std::string, std::string> meta;
	std::string line;
	bool header_seen = false, saw_data = false;
	int version = 0, line_no = 0;
	while(std::getline(in, line)) {
		++line_no;
		line = es_trim(line);
		if(line.empty()) continue;
		if(line[0] == '#') {
			const std::string comment = es_trim(line.substr(1));
			if(comment.compare(0, 16, "DNANM_ES_CHARGES") == 0) {
				if(header_seen || saw_data) throw oxDNAException("Duplicate or misplaced protein charge header at line %d", line_no);
				std::istringstream header(comment);
				std::string magic, extra;
				if(!(header >> magic >> version) || magic != "DNANM_ES_CHARGES" ||
						(header >> extra) || version != 1) {
					throw oxDNAException("Invalid protein charge format header at line %d", line_no);
				}
				header_seen = true;
			}
			else {
				const auto equal = comment.find('=');
				if(equal != std::string::npos) {
					const auto key = es_trim(comment.substr(0, equal));
					const auto value = es_trim(comment.substr(equal + 1));
					if(!meta.emplace(key, value).second) {
						throw oxDNAException("Duplicate charge metadata '%s'", key.c_str());
					}
				}
			}
			continue;
		}
		saw_data = true;
		const auto hash = line.find('#');
		if(hash != std::string::npos) line = es_trim(line.substr(0, hash));
		if(line.empty()) continue;

		ChargeEntry e;
		std::istringstream fields(line);
		std::string residue, charge_text, extra;
		if(version == 1) {
			if(!(fields >> e.local >> e.strand >> residue >> charge_text >> e.group) ||
					residue.size() != 1 || (fields >> extra)) {
				throw oxDNAException("Malformed version-1 protein charge record at line %d", line_no);
			}
			e.residue = static_cast<char>(std::toupper((unsigned char) residue[0]));
		}
		else {
			if(!(fields >> e.local >> charge_text >> e.group) || (fields >> extra)) {
				throw oxDNAException("Malformed legacy protein charge record at line %d", line_no);
			}
		}
		e.charge = es_finite_double(charge_text, "protein charge row");
		entries.push_back(e);
	}
	if(in.bad()) throw oxDNAException("I/O error reading protein charge file");

	const int npro = top.N_amino_acids;
	if(version == 1) {
		auto require = [&](const char *key) -> const std::string & {
			auto it = meta.find(key);
			if(it == meta.end()) throw oxDNAException("Charge file missing metadata '%s'", key);
			return it->second;
		};
		if(require("index_space") != "protein_local" || require("index_base") != "0" ||
				es_int(require("n_protein_beads"), "n_protein_beads") != npro) {
			throw oxDNAException("Charge file requires zero-based protein_local indices and matching n_protein_beads");
		}
	}
	else if(!meta.empty()) {
		throw oxDNAException("Protein charge metadata requires the version-1 header");
	}
	else {
		OX_LOG(Logger::LOG_WARNING, "Unversioned three-column protein charge file: no strand/residue validation");
	}

	if(entries.size() != expected.size()) {
		throw oxDNAException("Charge file has %d rows, expected %d",
				(int) entries.size(), npro);
	}

	_protein_charge_reduced.assign(top.N, (number) 0);
	_protein_group.assign(top.N, -1);
	std::vector<bool> seen(npro, false);
	std::set<int> groups;
	for(const auto &e : entries) {
		if(e.local < 0 || e.local >= npro || seen[e.local] || e.group < 0) {
			throw oxDNAException("Duplicate or invalid protein charge index/group at local bead %d", e.local);
		}
		if(version == 1 && (e.strand != expected[e.local].first ||
				e.residue != expected[e.local].second)) {
			throw oxDNAException("Protein charge strand or residue mismatch at local bead %d", e.local);
		}
		const int global = global_indices[e.local];
		const long double reduced =
				(static_cast<long double>(_protein_charge_scale) *
				static_cast<long double>(e.charge)) /
				static_cast<long double>(_dna_effective_charge_e);
		_protein_charge_reduced[global] = es_checked_number(reduced, "reduced protein charge");
		_protein_group[global] = e.group;
		seen[e.local] = true;
		groups.insert(e.group);
	}
	if(version == 1 && meta.count("n_groups") &&
			es_int(meta.at("n_groups"), "n_groups") != (int) groups.size()) {
		throw oxDNAException("Charge file n_groups does not match actual group count");
	}
	OX_LOG(Logger::LOG_INFO, "Loaded %d protein charges from %s (format %d)",
			npro, _protein_charge_file.c_str(), version);
}

number DNANMInteraction::_screened_pair(const LR_vector &r, number qproduct,
		LR_vector &force_on_second, bool update_forces) const {
	force_on_second = LR_vector(0, 0, 0);
	if(qproduct == 0) return (number) 0;

	const number distance = r.module();
	if(distance >= _debye_huckel_RC) return (number) 0;
	if(distance <= std::numeric_limits<number>::epsilon()) {
		throw oxDNAException("Overlapping charged beads in protein electrostatics");
	}

	number energy;
	number radial_force = 0;
	if(distance < _debye_huckel_RHIGH) {
		energy = qproduct * _debye_huckel_prefactor *
				std::exp(_minus_kappa * distance) / distance;
		if(update_forces) radial_force = -energy * (_minus_kappa - 1 / distance);
	}
	else {
		const number delta = distance - _debye_huckel_RC;
		energy = qproduct * _debye_huckel_B * SQR(delta);
		if(update_forces) radial_force = -2 * qproduct * _debye_huckel_B * delta;
	}
	if(update_forces) force_on_second = (r / distance) * radial_force;
	return energy;
}

number DNANMInteraction::_protein_dna_screened(BaseParticle *p, BaseParticle *q,
		bool compute_r, bool update_forces) {
	if(!_protein_electrostatics || _is_protein(p) == _is_protein(q)) return 0;
	if(compute_r) this->_computed_r = this->_box->min_image(p->pos, q->pos);
	BaseParticle *protein = _is_protein(p) ? p : q;
	BaseParticle *nucleotide = _is_protein(p) ? q : p;

	// rcentre = protein COM - nucleotide COM.
	const LR_vector rcentre = _is_protein(q) ? this->_computed_r : -this->_computed_r;
	const LR_vector rback = rcentre - nucleotide->int_centers[DNANucleotide::BACK];
	number qproduct = es_checked_number(
			-static_cast<long double>(_protein_dna_dh_scale) *
			static_cast<long double>(_protein_charge_reduced.at(protein->index)),
			"protein-DNA charge product");

	if(_debye_huckel_half_charged_ends &&
			(nucleotide->n3 == P_VIRTUAL || nucleotide->n5 == P_VIRTUAL)) {
		qproduct *= (number) 0.5;
	}
	LR_vector force_on_protein;
	const number energy = _screened_pair(rback, qproduct, force_on_protein, update_forces);
	if(update_forces && energy != 0) {
		protein->force += force_on_protein;
		nucleotide->force -= force_on_protein;
		nucleotide->torque += nucleotide->orientationT *
				nucleotide->int_centers[DNANucleotide::BACK].cross(-force_on_protein);
		this->_update_stress_tensor(nucleotide, protein, rcentre, force_on_protein);
	}
	return energy;
}

number DNANMInteraction::_protein_protein_screened(BaseParticle *p, BaseParticle *q,
		bool compute_r, bool update_forces) {
	if(!_protein_electrostatics || !_is_protein(p) || !_is_protein(q) || p->is_bonded(q)) return 0;
	if(_exclude_intragroup_electrostatics &&
			_protein_group.at(p->index) == _protein_group.at(q->index)) return 0;
	if(compute_r) this->_computed_r = this->_box->min_image(p->pos, q->pos);
	const number qproduct = es_checked_number(
			static_cast<long double>(_protein_protein_dh_scale) *
			static_cast<long double>(_protein_charge_reduced.at(p->index)) *
			static_cast<long double>(_protein_charge_reduced.at(q->index)),
			"protein-protein charge product");
	LR_vector force_on_q;
	const number energy = _screened_pair(this->_computed_r, qproduct, force_on_q, update_forces);
	if(update_forces && energy != 0) {
		p->force -= force_on_q;
		q->force += force_on_q;
		this->_update_stress_tensor(p, q, this->_computed_r, force_on_q);
	}
	return energy;
}

number DNANMInteraction::pair_interaction_nonbonded(BaseParticle *p, BaseParticle *q,
		bool compute_r, bool update_forces) {
	// The original ANM and oxDNA2 Hamiltonian is always evaluated first, unchanged.
	number energy = ANMHybridInteraction<DNA2Interaction>::pair_interaction_nonbonded(
			p, q, compute_r, update_forces);
	if(!_protein_electrostatics) return energy;

	// The shared parent can exit before the protein excluded-volume cutoff;
	// electrostatics must therefore be evaluated independently of that cutoff.
	if(_is_protein(p) != _is_protein(q)) {
		energy += _protein_dna_screened(p, q, compute_r, update_forces);
	}
	else if(_is_protein(p)) {
		energy += _protein_protein_screened(p, q, compute_r, update_forces);
	}
	return energy;
}
