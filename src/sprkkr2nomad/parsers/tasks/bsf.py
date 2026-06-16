"""NOMAD parser for SPRKKR BSF (Bloch spectral function) output files.

Handles two SPRKKR task variants:
  bsfek  — k-path (EK-REL) mode: A(E, k) band-structure-like spectral function
  bsfkk  — k-k plane (CONST-E) mode: A(k1, k2) constant-energy surface

The Bloch spectral function A(k, E) = -1/π Im[G^R(k, E)] is stored using
the ElectronicGreensFunction schema section from nomad_simulations.
"""

import numpy as np
from nomad.units import ureg
from nomad_simulations.schema_packages.outputs import Outputs
from nomad_simulations.schema_packages.properties import ElectronicGreensFunction
from nomad_simulations.schema_packages.variables import KMesh, Frequency

from .base import SprkkrTaskParser


class SprkkrBsfParser(SprkkrTaskParser, task_name='bsfek'):
    """Parser for BSF tasks (bsfek = k-path; bsfkk = k-k plane)."""

    def outputs(self, system):
        nomad = Outputs()
        nomad.model_system_ref = system

        bsf = self.output.bsf
        mode = bsf.MODE()
        keyword = bsf.KEYWORD()

        if mode == 'EK-REL':
            self._add_ekrel_sections(nomad, bsf, keyword)
        elif mode == 'CONST-E':
            self._add_conste_sections(nomad, bsf, keyword)

        return nomad

    # ------------------------------------------------------------------
    # EK-REL: A(E, k) — band-structure path
    # ------------------------------------------------------------------

    def _add_ekrel_sections(self, nomad, bsf, keyword):
        energies_eV = bsf.E()          # shape (NE,), eV relative to Fermi
        k_path = bsf.K()               # shape (NK,), dimensionless arc length

        if keyword == 'BSF':
            # spin-up and spin-down channels
            for spin_channel, data in [(0, bsf.I_UP()), (1, bsf.I_DOWN())]:
                # data shape: (NQ_EFF, NE, NK); sum over sites
                nomad.electronic_greens_functions.append(
                    self._ekrel_section(energies_eV, k_path,
                                        data.sum(axis=0), spin_channel)
                )
        else:
            # BSF-SPOL / BSF-SPN: spin-vector components + total
            for label, data in [
                ('total', bsf.I()),
                ('x',     bsf.I_X()),
                ('y',     bsf.I_Y()),
                ('z',     bsf.I_Z()),
            ]:
                gf = self._ekrel_section(energies_eV, k_path,
                                         data.sum(axis=0))
                gf.label = label
                nomad.electronic_greens_functions.append(gf)

    def _ekrel_section(self, energies_eV, k_path, data_ne_nk,
                       spin_channel=None):
        """Build one ElectronicGreensFunction for EK-REL data.

        Parameters
        ----------
        energies_eV : np.ndarray, shape (NE,)
        k_path      : np.ndarray, shape (NK,), dimensionless
        data_ne_nk  : np.ndarray, shape (NE, NK)
        spin_channel: int or None
        """
        gf = ElectronicGreensFunction()
        if spin_channel is not None:
            gf.spin_channel = spin_channel

        km = KMesh()
        km.points = k_path
        gf.k_mesh = km

        freq = Frequency()
        freq.points = energies_eV * ureg.eV
        gf.real_frequency = freq

        # Raw file values are in 1/Ry; normalise to 1/J via ureg
        gf.value = data_ne_nk / ureg.Ry

        return gf

    # ------------------------------------------------------------------
    # CONST-E: A(k1, k2) — constant-energy surface
    # ------------------------------------------------------------------

    def _add_conste_sections(self, nomad, bsf, keyword):
        k1 = bsf.K1()   # shape (NK1,)
        k2 = bsf.K2()   # shape (NK2,)

        # Build a flat (NK1*NK2, 2) array of k-point coordinates
        k1g, k2g = np.meshgrid(k1, k2, indexing='ij')
        k_points = np.stack([k1g.ravel(), k2g.ravel()], axis=-1)

        if keyword == 'BSF':
            for spin_channel, data in [(0, bsf.I_UP()), (1, bsf.I_DOWN())]:
                nomad.electronic_greens_functions.append(
                    self._conste_section(k_points, data.sum(axis=0),
                                         spin_channel)
                )
        else:
            for label, data in [
                ('total', bsf.I()),
                ('x',     bsf.I_X()),
                ('y',     bsf.I_Y()),
                ('z',     bsf.I_Z()),
            ]:
                gf = self._conste_section(k_points, data.sum(axis=0))
                gf.label = label
                nomad.electronic_greens_functions.append(gf)

    def _conste_section(self, k_points, data_nk1_nk2, spin_channel=None):
        """Build one ElectronicGreensFunction for CONST-E data.

        Parameters
        ----------
        k_points     : np.ndarray, shape (NK1*NK2, 2), dimensionless
        data_nk1_nk2 : np.ndarray, shape (NK1, NK2)
        spin_channel : int or None
        """
        gf = ElectronicGreensFunction()
        if spin_channel is not None:
            gf.spin_channel = spin_channel

        km = KMesh()
        km.points = k_points
        gf.k_mesh = km

        gf.value = data_nk1_nk2.ravel() / ureg.Ry

        return gf


# Register the same class for the k-k plane variant
SprkkrTaskParser._registry['bsfkk'] = SprkkrBsfParser
