"""Map SPRKKR spectral intensities to native NOMAD Green's-function sections.

The stored values retain the original parser convention: site-summed Bloch
spectral intensities A(E,k), not reconstructed complex Green's functions.
"""

import re
import warnings
from functools import cached_property
from pathlib import Path

import numpy as np
from ase.units import Bohr
from nomad.metainfo import Quantity, SubSection
from nomad.units import ureg
from nomad_simulations.schema_packages.numerical_settings import (
    KMesh as KMeshSettings,
)
from nomad_simulations.schema_packages.numerical_settings import (
    KSpace,
    Mesh,
    NumericalSettings,
)
from nomad_simulations.schema_packages.outputs import Outputs
from nomad_simulations.schema_packages.properties import ElectronicGreensFunction
from nomad_simulations.schema_packages.variables import Frequency, KMesh

from ...local_hdf5 import write_datasets
from .base import SprkkrTaskParser


class BsfSampling(NumericalSettings):
    """Preserve the original path distance or plane parameters alongside 3D k points."""

    parameters = SubSection(sub_section=Mesh.m_def)
    segment_labels = Quantity(type=str, shape=['*'])
    segment_end_indices = Quantity(
        type=np.int32, shape=['*'],
        description='One-based segment end indices from the BSF file (INDKDIR).',
    )


class SprkkrBsfParser(SprkkrTaskParser, task_name='bsfek'):
    """Read path (bsfek) and constant-energy plane (bsfkk) calculations."""

    def parse(self, archive):
        self._datasets = []
        super().parse(archive)
        # HDF5Dataset requires the archive context and final section paths.
        write_datasets(archive, self._datasets)

    @cached_property
    def sampling(self):
        bsf = self.output.bsf
        if bsf.MODE() == 'EK-REL':
            parameters = BsfSampling(
                name='SPRKKR BSF path distance (2*pi/a)',
                parameters=Mesh(
                    dimensionality=1, n_points=bsf.NK(),
                    points=np.asarray(bsf.K())[:, None],
                ),
                segment_labels=bsf.LBLKDIR(),
                segment_end_indices=bsf.INDKDIR(),
            )
            points = self._path_points(bsf)
            if points is None:
                warnings.warn(
                    'BSF path segments do not match the SPRKKR output; '
                    'preserving path distances without inventing 3D k points.',
                    stacklevel=2,
                )
                return parameters, None
        elif bsf.MODE() == 'CONST-E':
            k1, k2 = np.meshgrid(bsf.K1(), bsf.K2(), indexing='ij')
            parameters = BsfSampling(
                name='SPRKKR BSF plane parameters',
                parameters=Mesh(
                    dimensionality=2, n_points=k1.size,
                    grid=[bsf.NK1(), bsf.NK2()],
                    points=np.stack([k1.ravel(), k2.ravel()], axis=-1),
                ),
            )
            start = bsf.VECK_START()
            start = np.zeros(3) if start is None else np.asarray(start)
            points = (
                start
                + k1[..., None] * (np.asarray(bsf.VECK1()) - start)
                + k2[..., None] * (np.asarray(bsf.VECK2()) - start)
            ).reshape(-1, 3)
        else:
            raise ValueError(f'Unsupported BSF mode: {bsf.MODE()}')

        # SPRKKR Cartesian coordinates use 2*pi/a; NOMAD uses fractional
        # reciprocal coordinates. Both reciprocal bases contain 2*pi.
        potential = self.output.potential
        direct_basis = np.asarray(potential.atoms.cell) / (
            potential.LATTICE.ALAT() * Bohr
        )
        points = points @ direct_basis.T
        mesh = KMeshSettings(
            n_points=len(points), dimensionality=3, points=points,
        )
        return parameters, mesh

    def _path_points(self, bsf):
        """Use the printed KA/KE endpoints and INDKDIR counts per segment."""
        filename = getattr(self.output.output_file, 'name', self.output.output_file)
        text = Path(filename).read_text()
        labels = [''.join(str(label).split()) for label in bsf.LBLKDIR()]
        ends = np.asarray(bsf.INDKDIR(), dtype=int)
        if not len(ends) or len(labels) != len(ends) or ends[-1] != bsf.NK():
            raise ValueError('BSF segment boundaries do not match its k-point count')
        counts = np.diff(np.concatenate(([0], ends)))
        if np.any(counts < 1):
            raise ValueError('BSF path segments must contain at least one point')

        pattern = re.compile(
            r'^\s*([^\n(]+)\(([^()\n]+)\)\s*(?:\.\.\.|KE\s*=)'
            r'\s*\(([^()\n]+)\)\s*$',
            re.MULTILINE,
        )
        segments = []
        for match in pattern.finditer(text):
            label, start, end = match.groups()
            expected = labels[len(segments)]
            if expected:
                if ''.join(label.split()) != expected:
                    continue
            elif not label.strip().startswith('k-dir'):
                continue
            start = np.fromstring(start.replace('D', 'E'), sep=' ')
            end = np.fromstring(end.replace('D', 'E'), sep=' ')
            if start.shape != (3,) or end.shape != (3,):
                raise ValueError('A BSF path endpoint must have three coordinates')
            segments.append(np.linspace(start, end, counts[len(segments)]))
            if len(segments) == len(labels):
                return np.concatenate(segments)
        return None

    def model_method(self):
        method = super().model_method()
        method.numerical_settings.append(self.sampling[0])
        if self.sampling[1] is not None:
            method.numerical_settings.append(KSpace(
                name='SPRKKR BSF k points',
                k_mesh=[self.sampling[1]],
                reciprocal_lattice_vectors=(
                    2 * np.pi * self.output.potential.atoms.cell.reciprocal()
                    / ureg.angstrom
                ),
            ))
        return method

    def outputs(self, system):
        bsf = self.output.bsf
        mode = bsf.MODE()
        out = Outputs(model_system_ref=system)
        if mode == 'EK-REL':
            energies = np.asarray(bsf.E()) * ureg.eV
        elif mode == 'CONST-E':
            energies = np.array([bsf.ERYD()[0] - bsf.EFERMI()]) * ureg.Ry
        else:
            raise ValueError(f'Unsupported BSF mode: {mode}')

        if bsf.KEYWORD() == 'BSF':
            channels = [(None, 0, bsf.I_UP()), (None, 1, bsf.I_DOWN())]
        else:
            channels = [
                ('total', None, bsf.I()),
                ('x', None, bsf.I_X()),
                ('y', None, bsf.I_Y()),
                ('z', None, bsf.I_Z()),
            ]

        for label, spin, values in channels:
            gf = ElectronicGreensFunction(name='Bloch spectral function')
            if label is not None:
                gf.label = label
            if spin is not None:
                gf.spin_channel = spin
            gf.k_mesh = KMesh(n_points=self.sampling[0].parameters.n_points)
            if self.sampling[1] is not None:
                # Quantity references are assigned the owning section.
                gf.k_mesh.points = self.sampling[1]
            gf.real_frequency = Frequency(n_points=len(energies), points=energies)
            if mode == 'EK-REL':
                data = np.asarray(values).sum(axis=0)
            else:
                data = np.asarray(values).sum(axis=0).ravel()
            out.electronic_greens_functions.append(gf)
            self._datasets.append((gf, 'value', data / ureg.Ry))
        return out


SprkkrTaskParser._registry['bsfkk'] = SprkkrBsfParser
