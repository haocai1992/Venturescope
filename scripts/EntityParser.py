"""This script is for parsing the 2012 crunchbase dataset (from CMU), which is in JSON format."""

import pickle
import json

class EntityParser:
    """A class to read the JSON files in 2012 crunchbase dataset (from CMU)."""
    @staticmethod
    def LoadJsonEntity(filename):
        text = EntityParser.LoadStringEntityByFilename(filename)
        js = None
        try:
            if text:
                js = json.loads(text)
        except ValueError as e:
            print(e)
        return js

    @staticmethod
    def LoadStringEntityByFileHandler(fid):
        if fid:
            return pickle.load(fid)
        else:
            return ''

    @staticmethod
    def LoadStringEntityByFilename(filename, mode = 'rb'):
        fid = EntityParser.get_file_handler(filename, mode)
        obj = None
        if fid:
            try:
                obj = pickle.load(fid)
            except ValueError as e:
                obj = None
        return obj

    @staticmethod
    def get_file_handler(filename, mode):
        try:
            fid = open(filename, mode)
        except IOError:
            fid = None
        return fid

