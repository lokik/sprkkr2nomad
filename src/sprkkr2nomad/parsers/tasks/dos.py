"""NOMAD parser for SPRKKR DOS (density of states) output files."""

import numpy as np
from nomad.units import ureg
from nomad_simulations.schema_packages.outputs import Outputs
from nomad_simulations.schema_packages.properties.spectral_profile import (
    ElectronicDensityOfStates,
)
from nomad_simulations.schema_packages.variables import Energy2 as Energy

from .base import SprkkrTaskParser


class SprkkrDosParser(SprkkrTaskParser, task_name='dos'):
    # model_method() is inherited from SprkkrTaskParser — reads XC and
    # relativity from the converged potential, which is all DOS needs.

    def outputs(self, system):
        nomad = Outputs()
        nomad.model_system_ref = system

        dos_data = self.output.dos
        total = dos_data.total_dos()
        # dos shape: (n_spins, n_orbital_channels, n_energies)
        dos_array = np.asarray(total.dos)
        n_spins = dos_array.shape[0]
        # energy is already (E - E_F) * Rydberg in eV (see DOSOutputFile.energy)
        energies_eV = dos_data.energy

        for spin in range(n_spins):
            dos_section = ElectronicDensityOfStates()
            dos_section.spin_channel = spin

            energy_var = Energy()
            energy_var.points = energies_eV * ureg.eV
            dos_section.energies = energy_var

            # Sum over orbital (l) channels to get spin-resolved total DOS
            dos_section.value = dos_array[spin].sum(axis=0) / ureg.eV

            nomad.electronic_dos.append(dos_section)

        return nomad
