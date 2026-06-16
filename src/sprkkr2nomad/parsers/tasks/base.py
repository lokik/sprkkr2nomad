"""Base class for per-task NOMAD parsers of SPRKKR output files."""

import numpy as np
from nomad.units import ureg
from nomad_simulations.schema_packages.general import Simulation, Program
from nomad_simulations.schema_packages.model_method import (
    DFT, XCFunctional, XCComponent, RelativityModel,
)
from nomad_simulations.schema_packages.numerical_settings import SelfConsistency

from ..ase_atoms import ase_atoms_to_nomad_model_system
from ..input_parameters import model_method_section
from ..ase2sprkkr_to_nomad import nomad_section_from_sprkkr

# SPRKKR XC-POT keyword → LibXC canonical name
_XC_POT_TO_LIBXC = {
    'VWN':    'LDA_C_VWN',
    'MJW':    None,
    'VBH':    'LDA_C_VBH',
    'PBE':    'GGA_X_PBE',
    'PW92':   'GGA_X_PW91',
    'EV-GGA': 'GGA_X_EV93',
    'BJ':     'MGGA_X_BJ06',
    'MBJ':    'MGGA_X_BJ06',
}
_XC_POT_TO_LADDER = {
    'VWN': 'LDA', 'MJW': 'LDA', 'VBH': 'LDA',
    'PBE': 'GGA', 'PW92': 'GGA', 'EV-GGA': 'GGA',
    'BJ': 'metaGGA', 'MBJ': 'metaGGA',
}
_IREL_TO_LEVEL = {
    1: 'scalar',
    3: 'four-component',
}

_SPRKKR_URL = (
    'https://www.ebert.cup.uni-muenchen.de/old/index.php'
    '?option=com_content&view=article&id=8&catid=4&Itemid=7&lang=en'
)


class SprkkrTaskParser:
    """Base class for per-task NOMAD section builders.

    Subclasses declare which task they handle via the *task_name* keyword
    argument in the class statement:

        class SprkkrScfParser(SprkkrTaskParser, task_name='scf'): ...

    This causes them to be auto-registered in ``_registry`` so that
    ``for_output`` can look them up.
    """

    _registry: dict = {}

    def __init_subclass__(cls, task_name: str = None, **kwargs):
        super().__init_subclass__(**kwargs)
        if task_name is not None:
            SprkkrTaskParser._registry[task_name] = cls

    def __init__(self, output):
        self.output = output

    @classmethod
    def for_output(cls, output):
        """Return a parser instance for *output*, or ``None`` if unsupported."""
        task = getattr(output, 'task_name', None)
        klass = cls._registry.get(task)
        if klass is None:
            return None
        return klass(output)

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    def program(self):
        prog = Program()
        prog.name = 'SPRKKR'
        prog.version = self.output.program_info['version']
        return prog

    def model_system(self):
        ms = ase_atoms_to_nomad_model_system(self.output.potential.atoms)
        ms.datetime = self.output.program_info['start_time']
        return ms

    def _xc_from_potential(self, pot):
        """Build XCFunctional + jacobs_ladder string from potential SCF-INFO."""
        xc_pot_name = pot['SCF-INFO']['XC-POT']()
        libxc = _XC_POT_TO_LIBXC.get(xc_pot_name)
        ladder = _XC_POT_TO_LADDER.get(xc_pot_name)

        xc = XCFunctional()
        if libxc:
            xc.functional_key = libxc
            comp = XCComponent()
            comp.canonical_label = libxc
            comp.weight = 1.0
            xc.components.append(comp)

        return xc, ladder

    def _relativity_from_potential(self, pot):
        """Build RelativityModel from potential GLOBAL SYSTEM PARAMETER IREL."""
        irel = pot['GLOBAL SYSTEM PARAMETER']['IREL']()
        rel_model = RelativityModel()
        rel_model.level = _IREL_TO_LEVEL.get(irel, 'non-relativistic')
        return rel_model

    # ------------------------------------------------------------------
    # model_method: always reads from the potential (SCF-INFO + IREL)
    # and uses ip for input_parameters SubSection and SelfConsistency if
    # available. Subclasses may call super() and extend the result.
    # ------------------------------------------------------------------

    def model_method(self):
        """Build a DFT model-method section from the output potential.

        XC functional and relativistic treatment are read from the converged
        potential file (``SCF-INFO`` / ``GLOBAL SYSTEM PARAMETER``), which is
        available for every task type. If input parameters are present the
        task-specific ``input_parameters`` SubSection is also populated.
        """
        pot = self.output.potential
        ip = self.output.input_parameters

        # Use model_method_section to get the right DFT subclass (with
        # input_parameters SubSection definition) when ip is available,
        # otherwise fall back to plain DFT.
        if ip is not None:
            cls = model_method_section(ip)
        else:
            cls = DFT
        mm = cls()
        mm.name = 'KKR'
        mm.type = 'SPRKKR'
        mm.external_reference = _SPRKKR_URL

        if ip is not None:
            mm.input_parameters = nomad_section_from_sprkkr(
                cls.input_parameters.section.section_cls, ip
            )

        xc, ladder = self._xc_from_potential(pot)
        mm.xc = xc
        if ladder:
            mm.jacobs_ladder = ladder

        mm.contributions.append(self._relativity_from_potential(pot))

        nspin = pot['GLOBAL SYSTEM PARAMETER']['NSPIN']()
        mm.is_spin_polarized = nspin == 2

        return mm

    # ------------------------------------------------------------------
    # Stub – override in subclasses
    # ------------------------------------------------------------------

    def outputs(self, system):
        """Return an Outputs section for this task, or None."""
        return None

    # ------------------------------------------------------------------
    # Orchestration
    # ------------------------------------------------------------------

    def simulation(self):
        sim = Simulation()
        sim.program = self.program()
        sim.datetime = self.output.program_info['start_time']

        if mm := self.model_method():
            sim.model_method.append(mm)

        system = self.model_system()
        sim.model_system.append(system)

        if out := self.outputs(system):
            sim.outputs.append(out)

        return sim
