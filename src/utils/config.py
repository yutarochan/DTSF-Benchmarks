'''
Configuration Parser
Auxillary function to take a configuration file and append it to argument object.
Author: Yuya Jeremy Ong (yuyajeremyong@gmail.com)
'''
from __future__ import print_function
import ast
import configparser

import utils.utility as util

def parser(args):
    # Parse Configruation File
    config = configparser.ConfigParser()
    config.read(args.param)

    # Set Parameter to Argument Attributes
    for k, v in config.items('param'):
        # Automatic Type Casting
        if util.is_int(v): v = int(v)
        elif util.is_float(v): v = float(v)

        setattr(args, k, v)

    return args
