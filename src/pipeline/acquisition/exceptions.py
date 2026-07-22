'''
This file serves as a catalogue of known errors for use in scripts/ and gui/
'''

# Imports

# Constants

# Classes

class CameraError(Exception):
    '''
    Defines the parent class for the rest of these classes, doesn't care about the type of error
    '''

class CameraConnectionError(CameraError):
    '''
    Error raised when camera can't be found or opened at start()
    '''

class CameraTimeoutError(CameraError):
    '''
    Error raisd when a grab_one() call times out waiting for a frame
    '''

# Functions

#if __name__ == "__main__":
    