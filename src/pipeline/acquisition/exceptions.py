'''
This file serves as a catalogue of known errors for use in scripts/ and gui/
'''

# Imports

# Constants

# Classes

class CameraError(Exception):

    '''
    Defines the parent class for all acquisition-related errors.
    '''

class CameraConnectionError(CameraError):

    '''
    Error raised when camera can't be found or opened at start()
    '''

class CameraTimeoutError(CameraError):

    '''
    Error raised when a grab_one() call times out waiting for a frame
    '''
    def __innit__(self, timeout_ms: int):
        super().__init__(f"grab_one() timed out after {timeout_ms} ms")
        self.timeout_ms = timeout_ms

class CameraConfigurationError(CameraError):
    
    '''
    Error raised when a configuration value (eg. exposure, gain) is rejected by the camera.
    '''

    def __init__(self, parameter: str, value, reason: str = ""):
        message = f"invalid {parameter}={value}"
        if reason:
            message += f": {reason}"
        super().__init__(message)
        self.parameter = parameter
        self.value = value


# Functions

#if __name__ == "__main__":
    