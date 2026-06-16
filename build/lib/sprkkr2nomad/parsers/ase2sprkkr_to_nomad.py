""" Functions in this module are used to create NOMAD schema
form ASE2SPRKKR schema of input parameters """

from nomad.metainfo import Quantity, MEnum, SubSection
from nomad.datamodel import ArchiveSection

from ase2sprkkr.common.grammar_types import (
    TypedGrammarType,
    Keyword,
    BasicSeparator,
    Array,
    Range
)
from ase2sprkkr.common.alternative_types import numpy_types
from ase2sprkkr.common.value_definitions import ValueDefinition
from .class_utils import camelize, create_class


def nomad_class_name(definition):
    """ Returns the name that should be used for NOMAD class corresponding
    to a given definition """
    return camelize(definition.name) + 'SprkkrInputParametersSection'


def value_to_nomad(definition):
    """ Convert a ASE2SPRKKR value definition to a NOMAD one. """
    gt = definition.type
    description = definition.description(True)
    if definition.is_repeated.is_numbered:
        section = create_class(
                        nomad_class_name(definition),
                        (ArchiveSection,),
                        { 'index': Quantity(type=int),
                          'value': grammar_type_to_nomad(gt) },
                        description
                  )
        out = SubSection(sub_section=section, repeated=True)
    else:
        out = grammar_type_to_nomad(gt, description)
    return out


def grammar_type_to_nomad(gt, description=None, shape=None):
    """ Create a NOMAD value definition that is able to hold a given type.
    Returns None, if there is no sense to pass such value to NOMAD,
    e.g. if it is separator, or a fixed value
    """
    if isinstance(gt, TypedGrammarType):
        typ = gt.numpy_type
        if typ is object:
            typ = str
        if shape:
            typ = numpy_types[typ]
    elif isinstance(gt, Keyword):
        if len(gt.keywords) <= 1:
            return None
        typ = MEnum(*gt.keywords)
    elif isinstance(gt, BasicSeparator):
        return None
    elif isinstance(gt, Array):
        if gt.min_length == gt.max_length and gt.min_length is not None:
            shape = [ gt.min_length ]
        else:
            mlen = gt.max_length if gt.max_length is not None else "*"
            shape = [ f'{gt.min_length or 0}..{mlen}' ]
        return grammar_type_to_nomad(gt.type, description, shape)
    elif isinstance(gt, Range):
        return grammar_type_to_nomad(gt._type, description, [2])
    else:
        raise ValueError("This ASE2SPRKKR is not supported to be converted to NOMAD")
    return Quantity(type=typ, description=description, shape=shape)


def definition_to_nomad(definition):
    """
    Convert a ASE2SPRKKR value/section definition to NOMAD one.
    """
    if isinstance(definition, ValueDefinition):
        return value_to_nomad(definition)
    return section_to_nomad(definition)


def section_to_nomad(definition):
    """
    Convert ASE2SPRKKR section definition to NOMAD one.
    """

    def members():
        for i in definition:
            v = definition_to_nomad(i)
            if v:
                yield i.name.lower(), v

    childs = {i:v for i,v in members()}
    section = create_class(
                           nomad_class_name(definition),
                           (ArchiveSection,), childs,
                           definition.description()
              )
    return SubSection(sub_section = section)


def nomad_section_from_sprkkr(nomad_class, data):
    """ Create a NOMAD Section of a given class from given
    ASE2SPRKKR configuration section (e.g. input_parameters).
    The class could be created e.g. by :func:`section_to_nomad`.
    """

    # Use all values (explicit + defaults) so input_parameters are complete.
    dct = data.to_dict() or {}

    def normalize(value):
        if isinstance(value, dict):
            # Convert numbered repeated dicts to list of index/value pairs.
            if value and all(isinstance(k, int) for k in value.keys()):
                return [
                    {'index': k, 'value': normalize(v)}
                    for k, v in value.items()
                ]
            return {
                (k.lower() if isinstance(k, str) else k): normalize(v)
                for k, v in value.items()
            }
        if isinstance(value, list):
            return [normalize(v) for v in value]
        return value

    out = nomad_class.m_from_dict(normalize(dct))
    return out
