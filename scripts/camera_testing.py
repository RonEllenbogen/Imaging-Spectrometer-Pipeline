'''
This is a testing and minimal grab script to make sure we are able to grab images from the camera
'''

# Imports
from pypylon import pylon
import imageio.v3 as iio
from pathlib import Path

from pipeline.utils.helpers import load_config

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

def minimal_grab_script():
    """
    Grabs a camera frame, prints its shape and datatype, and saves as a TIFF.
    """

    # Get configs
    config = load_config("configs/default.yaml")
    serial = config["camera"]["serial_number"]
    auto_exposure = config["camera"]["auto_exposure"]
    auto_timeout = config["camera"]["auto_timeout"]
    exposure_time = config["camera"]["exposure_time"]
    timeout = config["camera"]["timeout"]

    # Get cameras
    tl_factory = pylon.TlFactory.GetInstance()
    devices = tl_factory.EnumerateDevices()

    # Open camera
    for d in devices:
        if d.GetSerialNumber() == serial:
            camera = pylon.InstantCamera(
                tl_factory.CreateDevice(d)
            )
            break
    camera.Open()

    print(camera.GetDeviceInfo().GetModelName())
    print(camera.GetDeviceInfo().GetFriendlyName())

    # Take image
    if auto_exposure:
        # Enable one-time auto exposure
        camera.ExposureAuto.SetValue("Once")

        # Start grabbing so the auto algorithm has images to work with
        camera.StartGrabbing()

        while camera.ExposureAuto.GetValue() != "Off":
            grab = camera.RetrieveResult(
                auto_timeout,
                pylon.TimeoutHandling_ThrowException
            )
            grab.Release()

        # Auto exposure has converged
        exposure = camera.ExposureTime.GetValue()
        print(f"Chosen exposure: {exposure:.0f} µs")

        # Grab the actual image
        grab = camera.RetrieveResult(
            auto_timeout,
            pylon.TimeoutHandling_ThrowException
        )

        image = grab.Array
        grab.Release()

        camera.StopGrabbing()

    else:
        camera.ExposureAuto.SetValue("Off")
        camera.ExposureTime.SetValue(exposure_time)

        print(f"Using exposure: {camera.ExposureTime.GetValue():.0f} µs")

        grab = camera.GrabOne(timeout)
        image = grab.Array
        grab.Release()

    # Old single-image grabbing code without exposure control
    '''
    # Grab image
    grab = camera.GrabOne(timeout)

    if grab.GrabSucceeded():
        image = grab.GetArray()
        print(image.shape)
        print(image.dtype)
    else:
        print("Image acquisition failed")

    grab.Release()
    '''
    # Save image
    output_path = Path(__file__).parent.parent / "data" / "raw" / "test_image.tiff"
    iio.imwrite(output_path, image)




if __name__ == "__main__":
   minimal_grab_script()