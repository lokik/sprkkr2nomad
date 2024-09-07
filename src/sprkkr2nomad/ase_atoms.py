""" Methods to transform ase atoms to NOMAD model """
from nomad_simulations.schema_packages.model_system import (
      AtomicCell,
      AtomsState,
      ModelSystem,
      Symmetry
    )

from nomad.metainfo import Quantity
import numpy as np
from ase2sprkkr import SPRKKRAtoms


class AtomsStateEx(AtomsState):
     """ An atomc site, that can has partial occupation """
     occupancy = Quantity(
        type=np.float64,
        default=1.0,
        description="""
        Value of the displacements applied to each atom in the simulation cell.
        """,
      )


def ase_atoms_to_nomad_model_system(atoms):
    """ Create NOMAD data for an ASE atoms objects """
    ms = ModelSystem()
    cell = ase_atoms_to_nomad_atomic_cell(atoms)
    ms.cell.append(cell)
    if sum(atoms.pbc) == 3:
        ms.symmetry.append(ase_atoms_to_nomad_symmetry(atoms, cell))
    if atoms.__class__.__name__ == 'SPRKKRAtoms' and atoms.regions:
        indices = np.arrange(len(atoms))
        for name, region in atoms.regions.items():
            a = region.create_atoms()
            ac = ase_atoms_to_nomad_model_system(a)
            ac.atoms_indices = indices[region.slice]
            ac.name = name
            ms.model_systems.append(ac)
    ms.dimensionality = 3
    return ms


def ase_atoms_to_nomad_symmetry(atoms, pointing_to):
    """ Create NOMAD symmetry data for an ASE atoms objects """
    symmetry = Symmetry()
    # ASE implementations of symmetry is a bit buggy and
    # it do not takes at account SPRKKR 'extras'
    SPRKKRAtoms.promote_ase_atoms(atoms)
    symmetry.bravais_lattice = atoms.cell.get_bravais_lattice().pearson_symbol
    atoms.space_group_number = atoms.spacegroup_info.number()
    symmetry.space_group_symbol = atoms.spacegroup_info.dataset['pointgroup']
    if pointing_to:
        symmetry.atomic_cell_ref = pointing_to
    return symmetry


def ase_atoms_to_nomad_atomic_cell(atoms):
    """ Create NOMAD cell data for an ASE atoms objects """
    cell = AtomicCell()
    cell.lattice_vectors = atoms.cell[:]
    cell.periodic_boundary_conditions = atoms.pbc

    atomic_numbers = atoms.get_atomic_numbers()

    def atoms_state(i, symbol, occ):
        a = AtomsStateEx()
        a.chemical_symbol = symbol
        a.atomic_number = atomic_numbers[i]
        return a

    if 'occupancy' in atoms.info:
        occ = atoms.info['occupancy']

        def sites():
            for i in range(len(atoms)):
                if i in occ:
                    for symbol, chance in occ[i].items():
                        yield i, symbol, chance
                else:
                    yield i, atoms.symbol[i], 1.0

        sites = [ i for i in sites() ]
        cell.n_atoms = len(sites)

        def distribute(array):
            out = np.empty((cell.n_atoms, ) + array.shape[1:], dtype=array.dtype)
            for i,j in enumerate(sites):
                out[i] = array[j[0]]
            return out

        cell.positions = distribute(atoms.positions)
        if atoms.__class__.__name__ == 'SPRKKRAtoms':
            cell.equivalent_atoms = distribute(atoms.spacegroup_info.
                                          equivalent_sites.mapping)
        cell.atoms_state = [ atoms_state(*atom) for atom in sites ]
    else:
        cell.n_atoms = len(atoms)
        cell.positions = atoms.positions
        cell.atoms_state = [ atoms_state(i,s, 1.) for i,s in enumerate(atoms.symbols) ]
        if atoms.__class__.__name__ == 'SPRKKRAtoms':
            cell.equivalent_atoms = atoms.spacegroup_info.equivalent_sites.mapping
    return cell
