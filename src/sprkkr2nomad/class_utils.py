""" Just a few utilities for working with classes and their names. """


def camelize(name: str):
    """ Make a given string camel-case """
    names = name.split('_')
    return ''.join((i.capitalize() for i in names))


def create_class(name, parents, members={}, doc=None, module=__name__, **kwargs):
    """ Create a class dynamically """
    out = type(name, parents, members, **kwargs)
    setup_class(out, module, doc)
    return out


def setup_class(cls, module=__name__, doc=None):
    """ Set up a few common attributes of a (dynamically created) class object """
    cls.__module__ = module
    cls.__doc__ = doc
    return cls
