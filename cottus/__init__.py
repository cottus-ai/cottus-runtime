import os
import sys

__version__ = "0.1.0"

#find _cottus_C in build directory for dev mode
if 'cottus' in sys.modules:
    _pkg_path = os.path.dirname(os.path.abspath(__file__))
    _root_path = os.path.dirname(_pkg_path)
    _build_path_debug = os.path.join(_root_path, 'build', 'cottus', 'csrc') 
    _build_path_release = os.path.join(_root_path, 'build') # CMake often puts it here
    
    if os.path.exists(_build_path_debug):
        sys.path.insert(0, _build_path_debug)
    if os.path.exists(_build_path_release):
        sys.path.insert(0, _build_path_release)

from .model import load_hf_model

try:
    import _cottus_C
    Engine = _cottus_C.Engine
    EngineConfig = _cottus_C.EngineConfig
except ImportError:
    #allow import for basic packaging even if compiled extension is missing
    Engine = None
    EngineConfig = None
