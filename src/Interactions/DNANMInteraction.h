/**
 * @brief Hybrid DNA/ANM model with optional CPU screened protein electrostatics.
 * Original ANM code: Jonah, Subhajit; refactor: Erik Poppleton (2026).
 * Screened electrostatics: Gioacchino Schifino (2026).
 */
#ifndef DNANM_INTERACTION_H
#define DNANM_INTERACTION_H

#include "DNA2Interaction.h"
#include "ANMHybridInteraction.h"
#include <string>
#include <vector>

class DNANMInteraction: public ANMHybridInteraction<DNA2Interaction> {
protected:
    BaseParticle *_new_nucleotide(int index, char kind) override;

    // New term indices must not overlap the nucleic-acid terms 0..7 or
    // the four shared ANM terms 8..11.
    enum { PRO_DNA_DH = 12, PRO_PRO_DH = 13 };

    bool _protein_electrostatics = false;
    bool _exclude_intragroup_electrostatics = true;
    number _protein_charge_scale = 1.0;
    number _protein_dna_dh_scale = 1.0;
    number _protein_protein_dh_scale = 1.0;
    number _dna_effective_charge_e = 0.815;
    std::string _protein_charge_file;

    // Indexed by global particle index, not protein-local index.
    std::vector<number> _protein_charge_reduced;
    std::vector<int> _protein_group;

    void _load_protein_charges(size_t n_particles);
    number _scaled_dh(const LR_vector &r, number qproduct,
                      LR_vector &force_on_second, bool update_forces);
    number _term_protein_dna_dh(BaseParticle *p, BaseParticle *q,
                               bool compute_r, bool update_forces);
    number _term_protein_protein_dh(BaseParticle *p, BaseParticle *q,
                                   bool compute_r, bool update_forces);

public:
    explicit DNANMInteraction(bool torsion);
    virtual ~DNANMInteraction();

    void get_settings(input_file &inp) override;
    void init() override;
    void read_topology(int *N_strands, std::vector<BaseParticle *> &particles) override;
    number pair_interaction_nonbonded(BaseParticle *p, BaseParticle *q,
                                     bool compute_r = true, bool update_forces = false) override;
};

#endif /* DNANM_INTERACTION_H */
