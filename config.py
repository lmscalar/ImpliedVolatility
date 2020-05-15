from configparser import RawConfigParser
from definitions import  CONFIG_DIR


import os



def config(filename='database.ini', section='sqlite'):

    configFilePath = os.path.join(CONFIG_DIR, filename)
    # create a parser
    parser = RawConfigParser()
    # read config file
    parser.read(configFilePath)

    # get section, default to postgresql
    db = {}
    if parser.has_section(section):
        params = parser.items(section)
        for param in params:
            db[param[0]] = param[1]
    else:
        raise Exception('Section {0} not found in the {1} file'.format(section, filename))

    return db