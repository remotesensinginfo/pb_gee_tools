from distutils.version import LooseVersion

PB_GEE_TOOLS_VERSION_MAJOR = 0
PB_GEE_TOOLS_VERSION_MINOR = 0
PB_GEE_TOOLS_VERSION_PATCH = 4

PB_GEE_TOOLS_VERSION = "{}.{}.{}".format(
    PB_GEE_TOOLS_VERSION_MAJOR,
    PB_GEE_TOOLS_VERSION_MINOR,
    PB_GEE_TOOLS_VERSION_PATCH,
)
PB_GEE_TOOLS_VERSION_OBJ = LooseVersion(PB_GEE_TOOLS_VERSION)
__version__ = PB_GEE_TOOLS_VERSION


PB_GEE_SEN1_ASCENDING = 1
PB_GEE_SEN1_DESCENDING = 2
