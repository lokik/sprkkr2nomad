""" Methods to transform ase atoms to NOMAD model """
from nomad_simulations.schema_packages.model_system import (
    ModelSystem,
    GlobalCrystalSymmetry,
    LocalCrystalSymmetry,
)
from nomad_simulations.schema_packages.atoms_state import AtomsState
from nomad.metainfo import Quantity
from nomad.units import ureg
import numpy as np
from ase2sprkkr import SPRKKRAtoms


class AtomsStateEx(AtomsState):
    """ An atomic site that can have partial occupation """
    occupancy = Quantity(
        type=np.float64,
        default=1.0,
        description="""
        Fractional occupancy of this atomic site.
        """,
    )


def ase_atoms_to_nomad_model_system(atoms):
    """ Create NOMAD data for an ASE atoms object """
    ms = ModelSystem()
    _populate_model_system(ms, atoms)
    if sum(atoms.pbc) == 3:
        ms.symmetry = ase_atoms_to_nomad_symmetry(atoms)
    if atoms.__class__.__name__ == 'SPRKKRAtoms' and atoms.regions:
        indices = np.arange(len(atoms))
        for name, region in atoms.regions.items():
            a = region.create_atoms()
            sub = ase_atoms_to_nomad_model_system(a)
            sub.particle_indices = indices[region.slice]
            sub.name = name
            ms.sub_systems.append(sub)
    ms.dimensionality = 3
    return ms


def _populate_model_system(ms, atoms):
    """ Fill positional/particle data directly onto a ModelSystem instance. """
    atomic_numbers = atoms.get_atomic_numbers()

    ms.lattice_vectors = atoms.cell[:] * ureg.angstrom
    ms.periodic_boundary_conditions = atoms.pbc

    def make_atoms_state(i, symbol, occ):
        a = AtomsStateEx()
        a.chemical_symbol = symbol
        a.atomic_number = int(atomic_numbers[i])
        a.occupancy = occ
        return a

    if 'occupancy' in atoms.info:
        occ_map = atoms.info['occupancy']

        def sites():
            for i in range(len(atoms)):
                if i in occ_map:
                    for symbol, chance in occ_map[i].items():
                        yield i, symbol, chance
                else:
                    yield i, atoms.symbols[i], 1.0

        site_list = list(sites())
        ms.n_particles = len(site_list)

        def distribute(array):
            out = np.empty((ms.n_particles,) + array.shape[1:], dtype=array.dtype)
            for k, (orig_idx, _sym, _occ) in enumerate(site_list):
                out[k] = array[orig_idx]
            return out

        ms.positions = distribute(atoms.positions) * ureg.angstrom
        ms.particle_states = [make_atoms_state(*atom) for atom in site_list]

        # Equivalent atoms
        if atoms.__class__.__name__ == 'SPRKKRAtoms':
            try:
                equiv = distribute(atoms.spacegroup_info.equivalent_sites)
                local_sym = LocalCrystalSymmetry()
                local_sym.equivalent_atoms = equiv.astype(np.int32)
                ms.local_symmetry = local_sym
            except Exception:
                pass
    else:
        ms.n_particles = len(atoms)
        ms.positions = atoms.positions * ureg.angstrom
        ms.particle_states = [
            make_atoms_state(i, s, 1.0)
            for i, s in enumerate(atoms.symbols)
        ]
        if atoms.__class__.__name__ == 'SPRKKRAtoms':
            try:
                equiv = np.array(atoms.spacegroup_info.equivalent_sites)
                local_sym = LocalCrystalSymmetry()
                local_sym.equivalent_atoms = equiv.astype(np.int32)
                ms.local_symmetry = local_sym
            except Exception:
                pass


# Pearson lattice letter → NOMAD lattice_type enum value
_PEARSON_LATTICE = {
    'a': 'a - triclinic',
    'm': 'm - monoclinic',
    'o': 'o - orthorhombic',
    't': 't - tetragonal',
    'h': 'h - hexagonal',
    'c': 'c - cubic',
}
# Pearson centering letter → NOMAD lattice_centering enum value
_PEARSON_CENTERING = {
    'P': 'P - primitive',
    'R': 'R - rhombohedral',
    'A': 'S - face centred',
    'B': 'S - face centred',
    'C': 'S - face centred',
    'S': 'S - face centred',
    'I': 'I - body centred',
    'F': 'F - all faces centred',
}


def ase_atoms_to_nomad_symmetry(atoms):
    """ Create NOMAD symmetry section for an ASE atoms object """
    sym = GlobalCrystalSymmetry()
    try:
        SPRKKRAtoms.promote_ase_atoms(atoms)
        sym.space_group_number = int(
            atoms.spacegroup_info.spacegroup_number()
        )
        # dataset.international gives the full H-M symbol, e.g. 'Im-3m'
        sym.space_group_symbol = atoms.spacegroup_info.dataset.international
    except Exception:
        pass
    try:
        pearl = atoms.cell.get_bravais_lattice().pearson_symbol  # e.g. 'cI'
        lat_letter, cen_letter = pearl[0], pearl[1]
        lat_type = _PEARSON_LATTICE.get(lat_letter)
        centering = _PEARSON_CENTERING.get(cen_letter)
        # 'hR' is rhombohedral (trigonal), not hexagonal
        if lat_letter == 'h' and cen_letter == 'R':
            lat_type = 'r - trigonal'
        if lat_type is not None:
            sym.lattice_type = lat_type
        if centering is not None:
            sym.lattice_centering = centering
    except Exception:
        pass
    return sym
