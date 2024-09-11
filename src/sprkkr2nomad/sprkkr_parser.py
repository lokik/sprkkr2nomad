""" This module contains functions, that creates NOMAD data schema
from SPRKKR output files, parsed by ASE2SPRKKR """

from typing import Dict
import re
from ase2sprkkr import TaskResult, InputParameters
# pip install git+https://github.com/nomad-coe/nomad-schema-plugin-simulation-workflow.git
# from simulationworkflowschema import SinglePoint
from nomad.datamodel import EntryArchive
from nomad.parsing.parser import MatchingParser
from nomad_simulations.schema_packages.general import Simulation, Program
from nomad_simulations.schema_packages.outputs import Outputs, SCFOutputs
from nomad_simulations.schema_packages.model_method import XCFunctional
from nomad_simulations.schema_packages.properties import FermiLevel, TotalEnergy
from nomad_simulations.schema_packages.numerical_settings import SelfConsistency

from .ase2sprkkr_to_nomad import nomad_section_from_sprkkr
from .ase_atoms import ase_atoms_to_nomad_model_system
from .input_parameters import model_method_section


def self_consistency(ip:InputParameters):
    sc = SelfConsistency()
    sc.scf_minimization_algorithm={
        'BROYDEN2': 'Broyden (second)',
        'ANDERSON': 'Anderson',
        'TCHEBY'  : 'Tchebyschev' }.get(ip.SCF.ALG())
    sc.n_max_iterations = ip.SCF.NITER()
    sc.threshold_change = ip.SCF.TOL()
    return sc


def model_method(ip:InputParameters):
    cls = model_method_section(ip)
    mm = cls()
    mm.label ='KKR'
    mm.type = 'SPRKKR'
    mm.reference = 'https://www.ebert.cup.uni-muenchen.de/old/index.php?option=com_content&view=article&id=8&catid=4&Itemid=7&lang=en'
    mm.input_parameters = nomad_section_from_sprkkr(
        cls.input_parameters.section.section_cls,
        ip
    )
    VXC = ip.SCF.VXC
    mm.jacobs_ladder = VXC.jacobs_ladder()

    xc = XCFunctional()
    xc.__class__.libxc_name =property(lambda self: self.__dict__['_libxc_name'])
    xc.__class__.libxc_name =xc.__class__.libxc_name.setter(lambda self, v: (breakpoint() or self.__dict__.__setitem__('_libxc_name', v)))
    xc.name = 'hybrid'
    xc.libxc_name = VXC.libxc_name()

    xc.weight = 1.0
    mm.xc_functionals.append(xc)

    rel = ip.MODE.MODE()
    if rel == 'SREL' or rel == 'SP-SREL':
        mm.relativity_method = 'scalar_relativistic'
    if rel == 'SP-SREL' or rel is None or rel == 'FREL':
        mm.is_spin_polarized = True
    else:
        mm.is_spin_polarized = False
    mm.core_electron_treatment = 'full all electron'
    if ip.task_name == 'scf':
        mm.numerical_settings.append(self_consistency(ip))
    return mm


def program(output):
    program = Program()
    program.name = 'SPRKKR'
    program.version = output.program_info['version']
    return program


def model_system(output):
    ms = ase_atoms_to_nomad_model_system(output.potential.atoms)
    ms.datetime = output.program_info['start_time']
    return ms


def scf_outputs(output, system):

    nomad = SCFOutputs()

    def set_scf_step(step, iteration):
        step.model_system_ref = system
        fl = FermiLevel()
        fl.value = iteration.energy.EF()
        fl.is_scf_converged = iteration.converged()
        step.fermi_levels.append(fl)

        te = TotalEnergy()
        te.value = iteration.energy.ETOT()
        te.is_scf_converged = iteration.converged()
        step.total_energies.append(fl)

    set_scf_step(nomad, output.iterations[-1])
    for i in output.iterations:
        scf_step = Outputs()
        set_scf_step(scf_step, i)
        nomad.scf_steps.append(scf_step)

    return nomad


def simulation(output):
    simulation = Simulation()
    simulation.program = program(output)
    simulation.method = 'KKR'
    simulation.datetime = output.program_info['start_time']
    simulation.model_method.append(model_method(output.input_parameters))
    system = model_system(output)
    simulation.model_system.append(system)
    if output.input_parameters.name == 'scf':
        simulation.outputs.append(scf_outputs(output, system))
    return simulation


class SprkkrParser(MatchingParser):

    match_task = re.compile(r" TASK\s+ = ([A-Z]+)\s+\n")

    def parse(
        self,
        mainfile: str,
        archive: EntryArchive,
        logger=None,
        child_archives: Dict[str, EntryArchive] = None,
    ) -> None:
        output = TaskResult.from_file(mainfile)
        sim = simulation(output)
        archive.data = sim
        x=sim.model_method[0]
        x.m_to_dict()
