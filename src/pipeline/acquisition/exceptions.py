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
    def __init__(self, timeout_ms: int):
        super().__init__(f"grab_one() timed out after {timeout_ms} ms")
        self.timeout_ms = timeout_ms

class CameraGrabError(CameraError):

    '''
    Error raised when a grab_one() call completes without timing out but
    still fails -- e.g. a GigE buffer underrun/incompletely-grabbed frame
    from dropped packets, or another transport-level fault reported by the
    backend. Deliberately its own subclass rather than a bare CameraError:
    unlike CameraConnectionError/CameraConfigurationError (which mean the
    device or its settings are actually broken), a single incomplete grab
    is typically a transient network hiccup -- the same camera and stream
    are still fine a frame later -- so CameraStream._run() tolerates a run
    of these the same way it already tolerates CameraTimeoutError, instead
    of killing the stream on the first occurrence.
    '''

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
    