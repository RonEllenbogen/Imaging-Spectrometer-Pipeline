'''
This is a testing and minimal grab script to make sure we are able to grab images from the camera
'''

# Imports
from pypylon import pylon

# Constants

# Classes

# Functions
def device_check():
    '''
    Checks and prints the available devices. No params or returns.
    '''

    devices = pylon.TlFactory.GetInstance().EnumerateDevices()
    #Loop through devices and print information
    for device in devices:
        print(device.GetFriendlyName(), device.GetModelName(), device.GetSerialNumber())



if __name__ == "__main__":
   device_check()