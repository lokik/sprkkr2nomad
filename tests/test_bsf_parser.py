"""BSF regression coverage using real files and a small non-square k plane."""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from ase.units import Bohr
from ase2sprkkr.output_files.output_files import CommonOutputFile
from ase2sprkkr.outputs.task_result import TaskResult
from nomad.config import config
from nomad.datamodel import EntryArchive, EntryMetadata
from nomad.parsing.parsers import match_parser, run_parser
from nomad.units import ureg
from nomad.utils import get_logger
from nomad_simulations.schema_packages.general import Simulation
from nomad_simulations.schema_packages.outputs import Outputs
from nomad_simulations.schema_packages.properties import ElectronicGreensFunction

from sprkkr2nomad.local_hdf5 import to_json_dict, write_datasets
from sprkkr2nomad.parsers.tasks.bsf import BsfSampling, SprkkrBsfParser

ROOT = Path(__file__).resolve().parents[1]
LOGGER = get_logger(__name__)
RY_JOULES = (1 * ureg.Ry).to('joule').magnitude
pytestmark = pytest.mark.filterwarnings(
    'error:.*not derived from its definition:SyntaxWarning'
)


@pytest.fixture(scope='module')
def source():
    return TaskResult.from_file(str(ROOT / 'examples/Fe_Bsf/Fe_BSFEK.out'))


def parse_bsf(source, bsf):
    output = SimpleNamespace(
        potential=source.potential, input_parameters=source.input_parameters,
        program_info=source.program_info, output_file=source.output_file, bsf=bsf,
    )
    archive = EntryArchive(metadata=EntryMetadata(mainfile=source.output_file.name))
    SprkkrBsfParser(output).parse(archive)
    return archive


@pytest.mark.parametrize('filename', ['Fe_BSFEK.out', 'Fe_BSF.out'])
def test_bsf_matching_and_json(filename, source):
    path = str(ROOT / 'examples/Fe_Bsf' / filename)
    parser, keys = match_parser(path)
    assert parser.name == 'sprkkr2nomad.parsers:sprkkr_parser'
    if filename == 'Fe_BSF.out':
        with pytest.warns(UserWarning, match='preserving path distances'):
            archive = run_parser(path, parser, keys, logger=LOGGER)[0]
    else:
        archive = run_parser(path, parser, keys, logger=LOGGER)[0]

    settings = archive.data.model_method[0].numerical_settings
    sampling = next(item for item in settings if isinstance(item, BsfSampling))
    np.testing.assert_array_equal(sampling.parameters.points[:, 0], source.bsf.K())
    np.testing.assert_array_equal(sampling.segment_end_indices, source.bsf.INDKDIR())

    native = archive.m_to_dict()
    exported = to_json_dict(archive)
    json.dumps(exported, allow_nan=False)
    properties = archive.data.outputs[0].electronic_greens_functions
    assert len(properties) == 2
    for spin, expected in enumerate([source.bsf.I_UP(), source.bsf.I_DOWN()]):
        ref = native['data']['outputs'][0]['electronic_greens_functions'][spin]['value']
        assert ref.startswith('/uploads/') and ref.endswith(f'/{spin}/value')
        value = exported['data']['outputs'][0]['electronic_greens_functions'][spin]['value']
        np.testing.assert_allclose(np.asarray(value) * RY_JOULES, expected.sum(axis=0))
        assert properties[spin].spin_channel == spin

    if filename == 'Fe_BSF.out':
        assert properties[0].k_mesh.points is None
    else:
        points = properties[0].k_mesh.points
        assert points.shape == (296, 3)
        # Back-convert to SPRKKR's Cartesian 2*pi/a coordinates. Check all
        # segment endpoints, including the discontinuous H -> N jump at 264.
        direct = np.asarray(source.potential.atoms.cell) / (
            source.potential.LATTICE.ALAT() * Bohr
        )
        cartesian = points @ np.linalg.inv(direct.T)
        np.testing.assert_allclose(
            cartesian[[0, 63, 64, 108, 109, 153, 154, 208, 209, 263, 264, 295]],
            [[0, 0, 0], [0, 0, 1], [0, 0, 1], [0, .5, .5],
             [0, .5, .5], [0, 0, 0], [0, 0, 0], [.5, .5, .5],
             [.5, .5, .5], [0, 0, 1], [0, .5, .5], [.5, .5, .5]],
            atol=1e-14,
        )
        assert cartesian[31, 2] == pytest.approx(31 / 63)
        assert '#/data/model_method/0/numerical_settings/' in (
            native['data']['outputs'][0]['electronic_greens_functions'][0]['k_mesh']['points']
        )


def test_spin_vector_components(source):
    bsf = CommonOutputFile.from_file(
        str(ROOT / 'examples/Fe_Bsf/Fe_BSF_BLOCHSF_spol.bsf')
    )
    archive = parse_bsf(source, bsf)
    exported = to_json_dict(archive.data)['outputs'][0]['electronic_greens_functions']
    for prop, label, expected in zip(
        exported, ['total', 'x', 'y', 'z'],
        [bsf.I(), bsf.I_X(), bsf.I_Y(), bsf.I_Z()], strict=True,
    ):
        assert prop['label'] == label
        np.testing.assert_allclose(
            np.asarray(prop['value']) * RY_JOULES, expected.sum(axis=0), atol=1e-14,
        )


@pytest.mark.parametrize('origin', [None, [.1, .2, .3]])
def test_constant_energy_plane(tmp_path, source, origin):
    path = tmp_path / 'plane.bsf'
    separator = '#' * 80
    header = (ROOT / 'examples/Fe_Bsf/Fe_BSF_BLOCHSF.bsf').read_text().split(separator)[0]
    header = header.replace('NE               200', 'NE                 1')
    path.write_text(header + f'''{separator}
#DATASET  plane
MODE      CONST-E
NK1              2
NK2              3
ERYD         0.80000   0.00100
{separator}
NK1              2
VECK1         1.1000    0.2000    0.3000
NK2              3
VECK2         0.1000    2.2000    0.3000
{separator}
''' + '\n'.join(str(value) for value in range(1, 19)) + '\n')
    bsf = CommonOutputFile.from_file(str(path), try_only='bsf', unknown=False)
    # ASE2SPRKKR exposes the plane origin as a non-stored field; supply the
    # explicit origin here and separately exercise the legacy zero default.
    if origin is not None:
        bsf.VECK_START = origin
    archive = parse_bsf(source, bsf)
    props = archive.data.outputs[0].electronic_greens_functions
    data = to_json_dict(archive.data)['outputs'][0]['electronic_greens_functions']
    np.testing.assert_allclose(np.array(data[0]['value']) * RY_JOULES, [1, 2, 3, 7, 8, 9])
    np.testing.assert_allclose(np.array(data[1]['value']) * RY_JOULES, [4, 5, 6, 10, 11, 12])
    assert props[0].real_frequency.points[0].to('Ry').magnitude == pytest.approx(
        .8 - bsf.EFERMI()
    )
    direct = np.asarray(source.potential.atoms.cell) / (
        source.potential.LATTICE.ALAT() * Bohr
    )
    expected = (
        [[0, 0, 0], [.05, 1.1, .15], [.1, 2.2, .3],
         [1.1, .2, .3], [1.15, 1.3, .45], [1.2, 2.4, .6]]
        if origin is None else
        [[.1, .2, .3], [.1, 1.2, .3], [.1, 2.2, .3],
         [1.1, .2, .3], [1.1, 1.2, .3], [1.1, 2.2, .3]]
    )
    np.testing.assert_allclose(props[0].k_mesh.points @ np.linalg.inv(direct.T), expected)


def test_local_hdf5_and_complex_json():
    archive = EntryArchive(data=Simulation(outputs=[Outputs(
        electronic_greens_functions=[ElectronicGreensFunction()]
    )]))
    gf = archive.data.outputs[0].electronic_greens_functions[0]
    original_paths = config.fs.staging, config.fs.public
    write_datasets(archive, [(gf, 'value', np.array([1 + 2j, 3 - 4j]) / ureg.Ry)])
    assert (config.fs.staging, config.fs.public) == original_paths
    value = to_json_dict(archive.data)['outputs'][0]['electronic_greens_functions'][0]['value']
    np.testing.assert_allclose(np.array(value['re']) * RY_JOULES, [1, 3])
    np.testing.assert_allclose(np.array(value['im']) * RY_JOULES, [2, -4])

    with pytest.raises(ValueError, match='Attach HDF5 quantities'):
        write_datasets(archive, [(ElectronicGreensFunction(), 'value', np.ones(2))])
    assert (config.fs.staging, config.fs.public) == original_paths
