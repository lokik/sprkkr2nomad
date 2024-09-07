""" This module contains functions, that creates NOMAD data schema
from SPRKKR output files, parsed by ASE2SPRKKR """

from typing import Dict
from nomad.datamodel import EntryArchive
import re
import os.path
from ..outputs.task_result import KkrProcess
from .ase_atoms import ase_atoms_to_nomad_model_system
# pip install git+https://github.com/nomad-coe/nomad-schema-plugin-simulation-workflow.git
# from simulationworkflowschema import SinglePoint
from nomad_simulations.schema_packages.general import Simulation, Program
from .ase2sprkkr_to_nomad import nomad_section_from_sprkkr
from nomad.parsing.parser import MatchingParser
from .input_parameters import model_method_section


def model_method(input_parameters):
    cls = model_method_section(input_parameters)
    mm = cls()
    mm.label ='KKR'
    mm.type = 'SPRKKR'
    mm.reference = 'https://www.ebert.cup.uni-muenchen.de/old/index.php?option=com_content&view=article&id=8&catid=4&Itemid=7&lang=en'
    mm.input_parameters = nomad_section_from_sprkkr(
        cls.input_parameters.section.section_cls,
        input_parameters
    )
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


def simulation(output, ):
    simulation = Simulation()
    simulation.program = program(output)
    simulation.method = 'KKR'
    simulation.datetime = output.program_info['start_time']
    simulation.model_method.append(model_method(output.input_parameters))
    simulation.model_system.append(model_system(output))
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
        logger.info('MyParser called')
        with open(mainfile, "rb") as f:
            raw_out = f.read()
            matches = self.match_task.search(raw_out.decode('utf8'))
            process = KkrProcess.class_for_task(matches[1])
            process = process(None, None, os.path.dirname(mainfile))
            f.seek(0)
            output=process.read_from_file(f)
        sim = simulation(output)
        archive.data = sim
        x=sim.model_method[0]
        x.m_to_dict()
