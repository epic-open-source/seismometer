# -*- coding: utf-8 -*-
# ---------------------------------
class Singleton(type):
    """
    Metaclass for implementing the single instance pattern.
    For any call to create an instance of the class,
    look up the unique instance to return, or generate a new instance.
    """

    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]

    def kill(cls):
        """
        Used if the single instance needs to be reset/removed (example unit tests).
        You will need to call the class again after kill to create a new instance.
        """
        if cls in cls._instances:
            del cls._instances[cls]
