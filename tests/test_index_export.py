"""Index views omit payloads by type, without losing scalar search fields."""

import json
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np
from nomad.datamodel import EntryArchive
from nomad.datamodel.hdf5 import HDF5Dataset
from nomad.metainfo import Quantity
from nomad.units import ureg
from nomad_simulations.schema_packages.general import Simulation
from nomad_simulations.schema_packages.outputs import Outputs
from nomad_simulations.schema_packages.properties import (
    ElectronicGreensFunction,
    TotalEnergy,
)
from nomad_simulations.schema_packages.variables import Frequency, KMesh

from sprkkr2nomad.indexing import to_index_dict
from sprkkr2nomad.local_hdf5 import to_json_dict, write_datasets

ROOT = Path(__file__).resolve().parents[1]


class EnergyWithMatrix(TotalEnergy):
    """An HDF5 quantity whose name is not `value`."""

    matrix = Quantity(type=HDF5Dataset, unit='joule')


def test_index_metadata_without_payload_reads(monkeypatch):
    energy = EnergyWithMatrix(value=-3 * ureg.eV)
    gf = ElectronicGreensFunction(
        name='Bloch spectral function', spin_channel=1,
        k_mesh=KMesh(n_points=3),
        real_frequency=Frequency(n_points=2, points=[-2, 4] * ureg.eV),
    )
    archive = EntryArchive(data=Simulation(outputs=[Outputs(
        electronic_greens_functions=[gf], total_energies=[energy],
    )]))
    write_datasets(archive, [
        (gf, 'value', np.ones((2, 3)) / ureg.Ry),
        (energy, 'matrix', np.eye(2, dtype=complex) * ureg.eV),
    ])

    def forbid_payload_read(*args, **kwargs):
        raise AssertionError('Index export must not read HDF5 dataset contents')

    with monkeypatch.context() as patch:
        patch.setattr(h5py.Dataset, '__getitem__', forbid_payload_read)
        indexed = to_index_dict(archive.data)
        filtered = to_index_dict(archive.data, exclude=lambda q, _: q.name == 'matrix')

    json.dumps(indexed, allow_nan=False)
    assert '/uploads/' not in json.dumps(indexed)
    prop = indexed['outputs'][0]['electronic_greens_functions'][0]
    assert 'value' not in prop
    assert prop['spin_channel'] == 1
    assert prop['data_summary'] == {
        'datasets': {'value': {'shape': [2, 3], 'dtype': 'float64', 'unit': '1 / joule'}},
        'axes': {
            'k_mesh': {'n_points': 3},
            'real_frequency': {
                'n_points': 2, 'unit': 'joule',
                'min': (-2 * ureg.eV).to('joule').magnitude,
                'max': (4 * ureg.eV).to('joule').magnitude,
            },
        },
    }
    scalar = indexed['outputs'][0]['total_energies'][0]
    assert scalar['value'] == (-3 * ureg.eV).to('joule').magnitude
    assert 'matrix' not in scalar
    assert scalar['data_summary']['datasets']['matrix'] == {
        'shape': [2, 2], 'dtype': 'complex128', 'unit': 'joule',
    }
    assert 'data_summary' not in filtered['outputs'][0]['total_energies'][0]

    # The indexing view does not mutate the archive or its ordinary exports.
    assert archive.m_to_dict()['data']['outputs'][0]['electronic_greens_functions'][0][
        'value'
    ].startswith('/uploads/')
    expanded = to_json_dict(archive.data)
    assert len(expanded['outputs'][0]['electronic_greens_functions'][0]['value']) == 2


def test_index_can_omit_unresolved_dataset_without_context():
    gf = ElectronicGreensFunction(name='Remote Green function')
    # Represent an already received reference without resolving or fetching it.
    gf.__dict__['value'] = '/uploads/remote/archive/entry#/data/value'
    data = to_index_dict(gf)
    assert data['name'] == 'Remote Green function'
    assert 'value' not in data
    assert data['data_summary'] == {'datasets': {'value': {'unit': '1 / joule'}}}


def test_data_view_defaults_to_index_json(tmp_path):
    destination = tmp_path / 'index.json'
    result = subprocess.run(
        [sys.executable, str(ROOT / 'scripts/nomad_data_view.py'),
         str(ROOT / 'examples/Fe_Bsf/Fe_BSFEK.out'), '--out', str(destination)],
        capture_output=True, text=True, timeout=120, check=True,
    )
    assert not result.stderr
    indexed = json.loads(destination.read_text())
    for prop in indexed['outputs'][0]['electronic_greens_functions']:
        assert 'value' not in prop
        assert prop['data_summary']['datasets']['value']['shape'] == [200, 296]
        assert prop['data_summary']['axes']['real_frequency']['n_points'] == 200
    assert destination.stat().st_size < 100_000
