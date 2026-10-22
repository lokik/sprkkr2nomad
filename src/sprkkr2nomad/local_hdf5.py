"""Local storage and JSON views for NOMAD's native HDF5 quantities."""

from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import RLock
from uuid import uuid4

import numpy as np
from nomad.config import config
from nomad.datamodel import EntryMetadata
from nomad.datamodel.context import ClientContext, ServerContext
from nomad.datamodel.hdf5 import HDF5Dataset

_STORAGE_LOCK = RLock()


class LocalHDF5Context(ClientContext):
    """Keep temporary archive HDF5 files alive for the archive's lifetime."""

    def __init__(self, mainfile, upload_id):
        super().__init__(local_dir=str(Path(mainfile).resolve().parent), upload_id=upload_id)
        self._storage = TemporaryDirectory(prefix='sprkkr2nomad-hdf5-')

    @contextmanager
    def storage(self):
        # NOMAD 1.4.2 resolves HDF5 paths via global UploadFiles settings.
        # Limit the override to the actual write, and restore it even on errors.
        with _STORAGE_LOCK:
            staging, public = config.fs.staging, config.fs.public
            try:
                config.fs.staging = str(Path(self._storage.name) / 'staging')
                config.fs.public = str(Path(self._storage.name) / 'public')
                yield
            finally:
                config.fs.staging, config.fs.public = staging, public


def write_datasets(archive, datasets):
    """Write (section, quantity name, value) triples after sections are attached.

    A server archive uses its existing storage. Standalone archives use local
    temporary files, without creating a processing Upload or connecting to a DB.
    """
    if not datasets:
        return

    if isinstance(archive.m_context, ServerContext):
        for section, name, value in datasets:
            setattr(section, name, value)
        return

    if not isinstance(archive.m_context, LocalHDF5Context):
        if archive.metadata is None:
            archive.metadata = EntryMetadata()
        if archive.metadata.upload_id is None:
            archive.metadata.upload_id = uuid4().hex
        if archive.metadata.entry_id is None:
            archive.metadata.entry_id = uuid4().hex
        archive.m_context = LocalHDF5Context(
            archive.metadata.mainfile or '.', archive.metadata.upload_id
        )

    from nomad.files import StagingUploadFiles

    with archive.m_context.storage():
        StagingUploadFiles(upload_id=archive.metadata.upload_id, create=True)
        for section, name, value in datasets:
            if section.m_root() is not archive:
                raise ValueError('Attach HDF5 quantities to the archive before writing')
            setattr(section, name, value)


def to_json_dict(section, **kwargs):
    """Export native archive structure with HDF5 datasets expanded to JSON arrays.

    The native ``m_to_dict`` keeps its HDF5 references; this data view expands
    them in the schema's declared units. Complex datasets use real/imaginary
    arrays, matching NOMAD's representation of complex numerical quantities.
    """
    def inline(quantity, owner, value, path):
        if isinstance(quantity.type, HDF5Dataset):
            dataset_value = getattr(owner, quantity.name)
            if isinstance(dataset_value, str):
                raise TypeError(f'Cannot expand unresolved HDF5 reference at {path}')
            with dataset_value as dataset:
                data = np.asarray(dataset[()])
            if np.iscomplexobj(data):
                return {'re': data.real.tolist(), 'im': data.imag.tolist()}
            return data.tolist()
        return value

    return section.m_to_dict(transform=inline, **kwargs)
