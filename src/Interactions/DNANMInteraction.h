/**
 * @brief Hybrid DNA/ANM Model
 *
 * Jonah, Subhajit Feb. 2026; restructured by Erik Nov. 2026 (see ANMHybridInteraction)
 *
 * To use the oxDNA2 model with ANM proteins, set
 *
 * interaction_type = DNANM (or DNACT to add bending and torsion along the protein backbone)
 *
 * in the input file
 *
 * Input options:
 *
 * @verbatim
 parfile = string (parameter file with the protein springs, set to none for DNA-only simulations; mandatory for DNACT)
 massfile = string (set massfile for simulations w/ different massed items)
 bending_k = float (DNACT only: global bending stiffness. If bending_k and torsion_k are not set they are read from the parameter file)
 torsion_k = float (DNACT only: global torsion stiffness)
 topology = new-style topology (proteins are type=peptide strands) or the legacy mixed format:
            <N> <N_strands> <ndna> <npro> <ndnas>
 @endverbatim
 */

#ifndef DNANM_INTERACTION_H
#define DNANM_INTERACTION_H

#include "DNA2Interaction.h"
#include "ANMHybridInteraction.h"

class DNANMInteraction: public ANMHybridInteraction<DNA2Interaction> {
protected:
	BaseParticle *_new_nucleotide(int index, char kind) override;

	// Protein charge data use global topology indices. Zero charge for nucleotides.
	bool _protein_electrostatics = false;
	bool _exclude_intragroup_electrostatics = true;
	number _protein_charge_scale = (number) 1.;
	number _protein_dna_dh_scale = (number) 1.;
	number _protein_protein_dh_scale = (number) 1.;
	number _dna_effective_charge_e = (number) 0.815;
	std::string _protein_charge_file;
	std::vector<number> _protein_charge_reduced;
	std::vector<int> _protein_group;

	void _read_protein_charges();

	// Returns force on the second site of r = second - first.
	number _screened_pair(const LR_vector &r, number qproduct,
					 LR_vector &force_on_second, bool update_forces) const;
	number _protein_dna_screened(BaseParticle *p, BaseParticle *q,
					 bool compute_r, bool update_forces);
	number _protein_protein_screened(BaseParticle *p, BaseParticle *q,
					 bool compute_r, bool update_forces);

public:
	// Protein-specific term IDs follow ANMHybridInteraction's 8..11.
	enum { PRO_DNA_SCREENED = 12, PRO_PRO_SCREENED = 13 };

	/// @param torsion use the ANMT protein model (DNACT)
	explicit DNANMInteraction(bool torsion);
	virtual ~DNANMInteraction();

	void get_settings(input_file &inp) override;
	void init() override;
	void read_topology(int *N_strands, std::vector<BaseParticle *> &particles) override;
	number pair_interaction_nonbonded(BaseParticle *p, BaseParticle *q,
					 bool compute_r = true, bool update_forces = false) override;

	bool protein_electrostatics_enabled() const override {
		return _protein_electrostatics;
	}
};

#endif /* DNANM_INTERACTION_H */
