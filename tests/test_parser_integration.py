"""Exercise installed plugins against real calculation outputs."""

import json
from pathlib import Path

import numpy as np
import pytest
from nomad.parsing.parsers import match_parser, run_parser
from nomad.utils import get_logger

ROOT = Path(__file__).resolve().parents[1]
LOGGER = get_logger(__name__)
pytestmark = pytest.mark.filterwarnings(
    "error:.*not derived from its definition:SyntaxWarning"
)


def parse_matched(path, expected_name):
    parser, keys = match_parser(str(path))
    assert parser is not None, f"No parser matched {path}"
    assert parser.name == expected_name
    archive = run_parser(str(path), parser, keys, logger=LOGGER)[0]
    assert archive.data is not None
    assert archive.data.model_system
    assert archive.data.outputs
    json.dumps(archive.m_to_dict(with_meta=False, resolve_references=False))
    return archive.data


def test_scf():
    data = parse_matched(
        ROOT / "examples/Fe_Scf/Fe_SCF.out",
        "sprkkr2nomad.parsers:sprkkr_parser",
    )
    assert data.program.name == "SPRKKR"
    assert data.model_method[0].relativity.level == "four-component"
    assert data.model_system[0].particle_states[0].chemical_symbol == "Fe"
    energy = data.outputs[0].total_energies[0].value.to("eV").magnitude
    assert np.isfinite(energy)
    assert energy < 0
    scf = data.outputs[0].scf_steps
    energies = scf.energies_total.to("eV").magnitude
    assert len(energies) > 1
    assert energies[-1] == pytest.approx(energy)
    np.testing.assert_allclose(
        scf.delta_energies_total.to("eV").magnitude,
        np.abs(np.diff(energies)),
    )


def test_dos():
    data = parse_matched(
        ROOT / "examples/Fe_Dos/Fe_DOS.out",
        "sprkkr2nomad.parsers:sprkkr_parser",
    )
    assert data.program.name == "SPRKKR"
    dos = data.outputs[0].electronic_dos
    assert len(dos) == 2
    for channel in dos:
        assert len(channel.value) == len(channel.energies.points)
        assert len(channel.value) > 1
        assert np.all(np.isfinite(channel.value.magnitude))


def test_simulation_parser_in_same_process():
    path = (
        ROOT.parent / "nomad-parser-plugins-simulation"
        / "tests/data/abinit/Fe/Fe.out"
    )
    if not path.is_file():
        pytest.skip("Requires the sibling simulation parser checkout's ABINIT sample")
    data = parse_matched(path, "nomad_simulation_parsers.parsers:abinit_parser")
    assert data.program.name == "ABINIT"
