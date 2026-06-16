"""NOMAD parser for SPRKKR SCF (self-consistent field) output files."""

import numpy as np
from nomad.units import ureg
from nomad_simulations.schema_packages.outputs import Outputs, SCFSteps
from nomad_simulations.schema_packages.properties import TotalEnergy, ChemicalPotential
from nomad_simulations.schema_packages.properties.energies import BaseEnergy
from nomad_simulations.schema_packages.properties.electronic_eigenvalues import Occupancy
from nomad_simulations.schema_packages.numerical_settings import SelfConsistency
from nomad_simulations.schema_packages.atoms_state import (
    ElectronicState, SphericalSymmetryState,
)

from .base import SprkkrTaskParser

# SPRKKR l-channel labels → angular momentum quantum number
_L_MAP = {'s': 0, 'p': 1, 'd': 2, 'f': 3}


class SprkkrScfParser(SprkkrTaskParser, task_name='scf'):

    def model_method(self):
        mm = super().model_method()

        ip = self.output.input_parameters
        if ip is not None:
            sc = SelfConsistency()
            sc.scf_minimization_algorithm = {
                'BROYDEN2': 'Broyden (second)',
                'ANDERSON': 'Anderson',
                'TCHEBY': 'Tchebyschev',
            }.get(ip.SCF.ALG())
            sc.n_max_iterations = ip.SCF.NITER()
            sc.threshold_change = ip.SCF.TOL()
            mm.numerical_settings.append(sc)

        return mm

    def outputs(self, system):
        output = self.output
        nomad = Outputs()
        nomad.model_system_ref = system

        last = output.iterations[-1]

        # --- Total energy (final iteration) ---------------------------------
        te = TotalEnergy()
        te.value = last.energy.ETOT() * ureg.eV  # ase2sprkkr stores ETOT in eV
        te.is_converged = last.converged()

        # Band energy as a named contribution to TotalEnergy
        for at in last.atomic_types.values():
            e_band = at.E_band() if hasattr(at, 'E_band') else None
            if e_band is not None:
                contrib = BaseEnergy()
                contrib.name = 'band'
                contrib.value = float(e_band.to('eV').v) * ureg.eV
                te.contributions.append(contrib)

        nomad.total_energies.append(te)

        # --- Fermi energy as chemical potential ------------------------------
        cp = ChemicalPotential()
        cp.type = 'electronic'
        cp.fermi_energy = last.energy.EF() * ureg.Ry
        nomad.chemical_potentials.append(cp)

        # --- Per-iteration SCF convergence arrays ----------------------------
        scf = SCFSteps()
        energies_total = []
        delta_potential_rms = []
        delta_energies_total = []
        durations = []
        ry_to_eV = float((1.0 * ureg.Ry).to('eV').magnitude)
        code_specific = {
            'iterations': [],
            'rms_b_error': [],
            'converged': [],
            'spin_moments': [],
            'orbital_moments': [],
            'energy_contour_min_eV': [],
            'semi_core_min_eV': [],
            'core_max_eV': [],
        }

        for iteration in output.iterations.values():
            etot_eV = float(iteration.energy.ETOT())
            energies_total.append(etot_eV * ureg.eV)

            # V-potential RMS error → delta_potential_rms (in joule via Ry)
            delta_potential_rms.append(iteration.error() * ureg.Ry)

            # B-field RMS error (captured by updated ase2sprkkr reader)
            b_err = iteration.b_error() if hasattr(iteration, 'b_error') else None
            code_specific['rms_b_error'].append(
                float(b_err) if b_err is not None else None
            )

            # Iteration duration in seconds (captured by updated reader)
            dur = iteration.duration() if hasattr(iteration, 'duration') else None
            if dur is not None:
                durations.append(float(dur) * ureg.s)

            code_specific['iterations'].append(iteration.iteration())
            code_specific['converged'].append(iteration.converged())
            code_specific['spin_moments'].append(iteration.moment.spin())
            code_specific['orbital_moments'].append(iteration.moment.orbital())
            code_specific['energy_contour_min_eV'].append(
                iteration.energy.EMIN() * ry_to_eV
            )
            escbot = (
                iteration.energy.ESCBOT()
                if hasattr(iteration.energy, 'ESCBOT') and iteration.energy.ESCBOT() is not None
                else None
            )
            code_specific['semi_core_min_eV'].append(
                escbot * ry_to_eV if escbot is not None else None
            )
            ectop = (
                iteration.energy.ECTOP()
                if hasattr(iteration.energy, 'ECTOP') and iteration.energy.ECTOP() is not None
                else None
            )
            code_specific['core_max_eV'].append(
                ectop * ry_to_eV if ectop is not None else None
            )

        # |ΔE_tot| between successive iterations (0 for the first)
        delta_energies_total.append(0.0 * ureg.eV)
        for e_prev, e_curr in zip(energies_total[:-1], energies_total[1:]):
            delta_energies_total.append(abs(e_curr - e_prev))

        scf.energies_total = energies_total
        scf.delta_potential_rms = delta_potential_rms
        scf.delta_energies_total = delta_energies_total
        if durations:
            scf.durations = durations
        scf.code_specific_quantities = code_specific
        nomad.scf_steps = scf

        # --- Per-l-channel occupancies (NOS from final iteration) ------------
        self._add_occupancies(nomad, system, last)

        return nomad

    # ------------------------------------------------------------------
    # Occupancy helper
    # ------------------------------------------------------------------

    def _add_occupancies(self, nomad, system, last_iteration):
        """Add per-l-channel Occupancy sections for each atomic type.

        For each inequivalent site type in the last iteration, creates an
        ElectronicState hierarchy on the matching AtomsState and registers
        Occupancy sections in the Outputs.
        """
        for at_type in last_iteration.atomic_types.values():
            symbol = at_type.symbol()
            orb_data = at_type.orbitals()  # numpy structured array
            if orb_data is None:
                continue

            # Find first particle_state for this element symbol
            atom_state = next(
                (ps for ps in system.particle_states
                 if getattr(ps, 'chemical_symbol', None) == symbol),
                None,
            )
            if atom_state is None:
                continue

            # Build parent ElectronicState with one sub_state per l-channel
            parent_es = ElectronicState()
            parent_es.name = f'{symbol} valence'
            atom_state.electronic_state = parent_es

            for row in orb_data:
                l_label = str(row['l']).strip()
                if l_label in ('sum', 'TOT', ''):
                    continue
                l = _L_MAP.get(l_label)
                if l is None:
                    continue

                nos = float(row['NOS'])

                child_es = ElectronicState()
                child_es.name = f'{symbol} {l_label}'
                sss = SphericalSymmetryState()
                sss.l_quantum_number = l
                child_es.spin_orbit_state = sss
                parent_es.sub_states.append(child_es)

                occ = Occupancy()
                occ.orbitals_state_ref = child_es
                occ.value = nos
                nomad.occupancies.append(occ)
