/*
 * ForceEnergy.cpp
 *
 *  Created on: Oct 1, 2013
 *      Author: rovigatti
 */

#include "ForceEnergy.h"

#include "../Forces/BaseForce.h"
#include "../Forces/MutualTrap.h"

#include <algorithm>
#include <map>
#include <typeinfo>
#include <utility>
#include <vector>

namespace {

std::map<BaseForce *, number> mutual_trap_energy_weights(
	const std::vector<BaseParticle *> &particles
) {
	std::map<std::pair<int, int>, std::vector<MutualTrap *>> pairs;
	for(auto p : particles) {
		for(auto force : p->ext_forces) {
			// A derived force may have different potential semantics.
			if(typeid(*force) != typeid(MutualTrap)) continue;
			auto trap = static_cast<MutualTrap *>(force);
			const int first = trap->particle_index();
			const int second = trap->reference_particle_index();
			if(first != p->index || first == second) continue;
			pairs[std::make_pair(std::min(first, second), std::max(first, second))].push_back(trap);
		}
	}

	std::map<BaseForce *, number> weights;
	for(const auto &pair : pairs) {
		const auto &traps = pair.second;
		std::vector<bool> grouped(traps.size(), false);
		for(size_t i = 0; i < traps.size(); ++i) {
			if(grouped[i]) continue;
			std::vector<MutualTrap *> protocol;
			size_t forward = 0, reverse = 0;
			for(size_t j = i; j < traps.size(); ++j) {
				if(grouped[j] || !traps[i]->same_energy_protocol(*traps[j])) continue;
				grouped[j] = true;
				protocol.push_back(traps[j]);
				if(traps[j]->particle_index() == pair.first.first) ++forward;
				else ++reverse;
			}
			// Only a balanced set of equivalent reciprocal declarations
			// demonstrably duplicates one physical pair energy per pair.
			// Unpaired, mismatched or imbalanced declarations keep weight 1.
			if(forward > 0 && forward == reverse) {
				for(auto trap : protocol) weights[trap] = (number) 0.5;
			}
		}
	}
	return weights;
}

} // namespace

ForceEnergy::ForceEnergy() {

}

ForceEnergy::~ForceEnergy() {

}

void ForceEnergy::get_settings(input_file &my_inp, input_file &sim_inp) {
	BaseObservable::get_settings(my_inp, sim_inp);

	getInputString(&my_inp, "print_group", _group_name, 0);
	getInputBool(&my_inp, "per_particle", &_per_particle, 0);
}

std::string ForceEnergy::get_output_string(llint curr_step) {
	// Determine ownership on the complete inventory before group filtering:
	// the two endpoint shares remain additive even in different print groups.
	const auto weights = mutual_trap_energy_weights(_config_info->particles());
	number U = (number) 0.f;
	for(auto p: _config_info->particles()) {
		if(_group_name == "") {
			p->set_ext_potential(curr_step, _config_info->box);
			U += p->ext_potential;
			// Preserve the legacy per-particle potential cache and all other
			// force types; correct only the observable's duplicated pair sum.
			LR_vector abs_pos = _config_info->box->get_abs_pos(p);
			for(auto ext_force : p->ext_forces) {
				const auto weight = weights.find(ext_force);
				if(weight != weights.end()) {
					ext_force->set_current_particle(p);
					U -= ((number) 1. - weight->second) * ext_force->potential(curr_step, abs_pos);
				}
			}
		}
		else {
			LR_vector abs_pos = _config_info->box->get_abs_pos(p);
			for(auto ext_force : p->ext_forces) {
				if(ext_force->get_group_name() == _group_name) {
					ext_force->set_current_particle(p);
					const auto weight = weights.find(ext_force);
					const number share = weight == weights.end() ? (number) 1. : weight->second;
					U += share * ext_force->potential(curr_step, abs_pos);
				}
			}
		}
	}
	if(_per_particle) {
		U /= _config_info->N();
	}

	return Utils::sformat(_number_formatter, U);
}
