"""Archive views for indexing, without HDF5Dataset payloads or references."""

import numpy as np
from nomad.datamodel.hdf5 import HDF5Dataset
from nomad.metainfo import QuantityReference


def _is_hdf5(definition):
    return isinstance(getattr(definition, 'type', None), HDF5Dataset)


def _dataset_summary(quantity, owner):
    summary = {}
    if quantity.unit is not None:
        summary['unit'] = str(quantity.unit)
    value = getattr(owner, quantity.name)
    if isinstance(value, str):
        # A remote/unresolved reference is omitted too; do not fetch its data.
        return summary
    try:
        with value as dataset:
            summary['shape'] = list(dataset.shape) if dataset.shape is not None else None
            summary['dtype'] = str(dataset.dtype)
            if 'unit' not in summary and (unit := dataset.attrs.get('units')):
                summary['unit'] = unit.decode() if isinstance(unit, bytes) else str(unit)
    except (OSError, KeyError):
        # Indexing does not require the external dataset to still be accessible.
        pass
    return summary


def _axes_summary(owner, selected):
    axes = {}
    for name, definition in owner.m_def.all_sub_sections.items():
        if definition.repeats or not selected(definition, owner):
            continue
        axis = getattr(owner, name)
        if axis is None or 'points' not in axis.m_def.all_quantities:
            continue
        summary = {}
        count = getattr(axis, 'n_points', None)
        if count is not None:
            summary['n_points'] = int(count)
        points_definition = axis.m_def.all_quantities['points']
        if not selected(points_definition, axis):
            if summary:
                axes[name] = summary
            continue
        if points_definition.unit is not None:
            summary['unit'] = str(points_definition.unit)
        # Never load HDF5 axes or follow references to potentially large grids.
        if not _is_hdf5(points_definition) and not isinstance(
            points_definition.type, QuantityReference
        ):
            points = axis.points
            if points is not None:
                if hasattr(points, 'magnitude'):
                    if points_definition.unit is not None:
                        points = points.to(points_definition.unit)
                    else:
                        summary['unit'] = str(points.units)
                    points = points.magnitude
                points = np.asarray(points)
                if points.ndim == 1:
                    summary.setdefault('n_points', len(points))
                    if points.size and points.dtype.kind in 'iuf' and np.isfinite(points).all():
                        summary.update(min=float(points.min()), max=float(points.max()))
        if summary:
            axes[name] = summary
    return axes


def to_index_dict(section, **kwargs):
    """Omit HDF5Dataset quantities and add small per-section data summaries.

    ``data_summary.datasets`` records shape, dtype and units when available;
    ``data_summary.axes`` records axis counts and real-valued ranges. Metadata
    inspection never reads dataset contents. Ordinary quantities, including
    scalar ``value`` fields, retain their native serialization.
    """
    include = kwargs.pop('include', None)
    exclude = kwargs.pop('exclude', None)
    if include is not None and exclude is not None:
        raise ValueError('Use either include or exclude, not both')

    def selected(definition, owner):
        if include is not None:
            return include(definition, owner)
        return exclude is None or not exclude(definition, owner)

    data = section.m_to_dict(
        exclude=lambda definition, owner: (
            _is_hdf5(definition) or not selected(definition, owner)
        ),
        **kwargs,
    )

    def summarize(owner, target):
        datasets = {
            name: _dataset_summary(quantity, owner)
            for name, quantity in owner.m_def.all_quantities.items()
            if _is_hdf5(quantity) and owner.m_is_set(quantity) and selected(quantity, owner)
        }
        if datasets:
            summary = {'datasets': datasets}
            if axes := _axes_summary(owner, selected):
                summary['axes'] = axes
            target['data_summary'] = summary

        for name, definition in owner.m_def.all_sub_sections.items():
            if name not in target:
                continue
            children = getattr(owner, name)
            serialized = target[name]
            if definition.repeats:
                targets = serialized.values() if isinstance(serialized, dict) else serialized
                for child, child_data in zip(children, targets, strict=True):
                    if child is not None and isinstance(child_data, dict):
                        summarize(child, child_data)
            elif children is not None and isinstance(serialized, dict):
                summarize(children, serialized)

    summarize(section, data)
    return data
