""" Functions in this module create NOMAD data schema from
ASE2SPRKKR input parameters """

from nomad_simulations.schema_packages.model_method import ModelMethod
from .ase2sprkkr_to_nomad import section_to_nomad
from .class_utils import setup_class
import sys
import types
from ase2sprkkr.input_parameters.input_parameters import InputParameters


class SprkkrModelMethodMeta(ModelMethod.__class__):
    """ All SPRKKR model methods will have input_parameters
    property, added here. However, the properties can differ
    according to the task. This meta_class manage that the
    class will have the right data schema."""

    def __new__(self, cls_name, bases, members, /, input_parameters_definition=None):
        if input_parameters_definition:
            members['input_parameters'] = section_to_nomad(input_parameters_definition)
        cls = ModelMethod.__class__.__new__(self, cls_name, bases, members)
        return cls

    def __call__(self, *args, **kwargs):
        if not hasattr(self, 'input_parameters'):
            raise NotImplementedError("SprkkrModelMethod have to have InputParameters specified")
        return super().__call__(*args, **kwargs)


class SprkkrModelMethod(ModelMethod, metaclass=SprkkrModelMethodMeta):
     """ SPRKKR model method has additional input_parameters property.
     Meta class :class:`SprkkrModelMethodMeta` add them to the descendants
     of the class """


def model_method_name(task):
    """ Return the name of the NOMAD section for a given
    input parameters """
    name = task.capitalize()
    return f'Sprkkr{name}ModelMethod'


def model_method_section(task):
    """ Return the NOMAD section for a given parameter """
    if not isinstance(task, str):
        task = task.name
    return getattr(sys.modules[__name__], model_method_name(task))


def create_model_class(input_parameters_definition):
    """ Create NOMAD section for given ASE2SPRKKR input_parameters_definition class """
    if isinstance(input_parameters_definition, str):
        input_parameters_definition = InputParameters.task_definition(
            input_parameters_definition
        )
    name = input_parameters_definition.name
    cls = types.new_class(
           model_method_name(name),
           (SprkkrModelMethod, ),
           { 'input_parameters_definition' : input_parameters_definition }
    )
    return setup_class(
                 cls,
                 doc = f'Description of SPRKKR computation of {name.upper()}',
                 module = __name__,
    )


""" Now, lets redefine the module to create the nomad
representation of sprkkr input-parameters in a lazy way """


class InputParametersModule(types.ModuleType):

    """ This module will have the NOMAD models for each SPRKKR task
    defined in ASE2SPRKKR. However, they are created in a lazy way to
    be more efficient """

    def __getattr__(self, name):
        # Dynamically create the definitions as they are needed
        if name.startswith("Sprkkr") and name.endswith("ModelMethod"):
            task_name = name[6:-11]
            out = create_model_class(task_name)
            globals()[out.__name__] = out
            return out
        if name == '__all__':
            self.__all__ = [ model_method_name(i) for i in InputParameters.definitions.keys()]
            return self.__all__
        raise AttributeError(f"There is no such attribute {name} in the module.")


sys.modules[__name__].__class__ = InputParametersModule
