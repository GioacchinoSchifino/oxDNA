/**
 * @file    MutualTrap.h
 * @date    18/oct/2014
 * @author  Flavio 
 *
 */

#ifndef MUTUALTRAP_H_
#define MUTUALTRAP_H_

#include "BaseForce.h"

class MutualTrap: public BaseForce {
private:
	int _particle;
	int _ref_id;

public:
	BaseParticle * _p_ptr;
	number _r0;
	number _rate;
	number _stiff_rate;
	bool PBC;

	MutualTrap();
	virtual ~MutualTrap() {
	}

	std::tuple<std::vector<int>, std::string> init(input_file &inp) override;

	virtual LR_vector force(llint step, LR_vector &pos);
	virtual number potential(llint step, LR_vector &pos);

	// Read-only metadata for observable accounting; force/potential stay intact.
	int particle_index() const { return _particle; }
	int reference_particle_index() const { return _ref_id; }
	bool same_energy_protocol(const MutualTrap &other) const {
		return _r0 == other._r0 && _stiff == other._stiff &&
			_rate == other._rate && _stiff_rate == other._stiff_rate &&
			PBC == other.PBC;
	}

protected:
	LR_vector _distance(LR_vector u, LR_vector v);
};

#endif // MUTUALTRAP_H
