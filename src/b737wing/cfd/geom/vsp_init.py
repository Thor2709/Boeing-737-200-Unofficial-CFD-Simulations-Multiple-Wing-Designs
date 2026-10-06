import os
import sys

from b737wing.config import B737_OPENVSP_DIR

OPENVSP = str(B737_OPENVSP_DIR)
sys.path[:0] = [os.path.join(OPENVSP, r'python\openvsp'), os.path.join(OPENVSP, r'python\openvsp_config')]
if hasattr(os, "add_dll_directory") and os.path.isdir(OPENVSP):
    try:
        os.add_dll_directory(OPENVSP)
        sub_dll = os.path.join(OPENVSP, r'python\openvsp\openvsp')
        if os.path.isdir(sub_dll):
            os.add_dll_directory(sub_dll)
    except OSError:
        pass
os.environ['PATH'] = OPENVSP + ';' + os.path.join(OPENVSP, r'python\openvsp\openvsp') + ';' + os.environ.get('PATH', '')
import openvsp_config
openvsp_config.LOAD_GRAPHICS = False
openvsp_config.LOAD_FACADE = False
openvsp_config._IGNORE_IMPORTS = True
import openvsp as vsp
