import os
from typing import Optional, List

import cv2
from pypylon import genicam, pylon

from pe.utils.camera_utils import Intrinsic
from glob import glob
import numpy as np


class ImageLoader:
    def __init__(
        self,
        image_path: str,
        ext: str = "png",
        loop: bool = False,
    ) -> None:
        self.loop = loop
        self.ext = ext
        self.data = self._get_images_path(image_path)
        self.idx = 0
        self.intrinsic = None

        img = cv2.imread(self.data[0])
        self.height, self.width = img.shape[:2]

    def __iter__(self):
        self.idx = 0
        return self

    def __next__(self):
        if self.idx >= len(self.data):
            self.idx = 0
            if not self.loop:
                raise StopIteration
        rgb_data = self.data[self.idx]
        rgb = cv2.imread(rgb_data)

        ret = True if isinstance(rgb, np.ndarray) else False
        self.idx += 1
        return ret, rgb

    def _get_images_path(self, path: str) -> List[str]:
        if not os.path.isdir(path):
            raise RuntimeError("Path is not a directory")
        images_path = sorted(glob(os.path.join(path, f"*.{self.ext}")))
        if len(images_path) == 0:
            raise RuntimeError("No images found")

        return images_path

    def get(self):
        return next(self)

    def get_image_size(self):
        return self.width, self.height

    def load_intrinsic(self, intrinsic_file):
        self.intrinsic = Intrinsic.load(intrinsic_file)


class Basler:
    def __init__(self, model=None, device_class=None, ip_address=None ,intrinsic: Optional[Intrinsic] = None):

        if device_class is None:
            # conecting to the first available camera
            self.camera = pylon.InstantCamera(
                pylon.TlFactory.GetInstance().CreateFirstDevice()
            )
            self.model = self.camera.GetDeviceInfo().GetModelName()
            self._calss = self.camera.GetDeviceInfo().GetDeviceClass()
        elif device_class == 'BaslerGigE':
            tlFactory = pylon.TlFactory.GetInstance()
            devices = tlFactory.EnumerateDevices()
            for device in devices:
                if device.GetIpAddress==ip_address:
                    self.camera = pylon.InstantCamera(tlFactory.CreateDevice(device))
                    self.model = device.GetModelName()
                    self._class = device.GetDeviceClass()
                    print("Camera model: ", self.model)
                    print("Device Class: ", self._class)
        else:
            tlFactory = pylon.TlFactory.GetInstance()
            devices = tlFactory.EnumerateDevices()
            print(len(devices))
            for device in devices:
                if device.GetModelName() == model:
                    self.camera = pylon.InstantCamera(tlFactory.CreateDevice(device))
                    self.model = model
                    self._class = device.GetDeviceClass()
       

        self.intrinsic = intrinsic

        # Grabing Continusely (video) with minimal delay
        self.camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
        self.converter = pylon.ImageFormatConverter()

        # converting to opencv bgr format
        self.converter.OutputPixelFormat = pylon.PixelType_BGR8packed
        self.converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned

        while self.camera.IsGrabbing():
            grabResult = self.camera.RetrieveResult(
                5000, pylon.TimeoutHandling_ThrowException
            )
            if grabResult.GrabSucceeded():
                # Access the image data
                image = self.converter.Convert(grabResult)
                img = image.GetArray()
                h, w, _ = img.shape
                print("Image shape: {}x{}".format(w, h))
                break
            grabResult.Release()

        self.width = w
        self.height = h

    def load_intrinsic(self, intrinsic_file):
        self.intrinsic = Intrinsic.load(intrinsic_file)

    def get(self):
        img = None
        ret = True
        for i in range(10):
            grabResult = self.camera.RetrieveResult(
                5000, pylon.TimeoutHandling_ThrowException
            )
            if grabResult.GrabSucceeded():
                # Access the image data
                image = self.converter.Convert(grabResult)
                img = image.GetArray().copy()
                grabResult.Release()
                break
            print(self.model, i)
            grabResult.Release()

        if img is None:
            print("[!] Basler Grab Fail")
            print(self.isGrabbing())
            ret = False

        return ret, img

    def isGrabbing(self):
        return self.camera.IsGrabbing()

    def setConfig(self, gain=None, exposureTime=None):
        """
        cam.setConfig(**_config['config'])
        """
        if gain is not None:
            if self._class == "BaslerUsb":
                self.camera.Gain.Value(gain)
            else:  # 'BaslerGigE'
                self.camera.GainRaw.Value(gain)
        if exposureTime is not None:
            if self._class == "BaslerUsb":
                self.camera.ExposureTime.Value(exposureTime)
            else:
                self.camera.ExposureTimeAbs.Value(80000.0)
        self.camera.BalanceWhiteAuto.Value("Once")


if __name__ == "__main__":
    import argparse
    from datetime import datetime

    parser = argparse.ArgumentParser(description="camera")
    parser.add_argument(
        "--out-dir",
        "-o",
        default=".",
        help="path to save dir",
    )
    parser.add_argument(
        "--intrinsic",
        "-i",
        default="intrinsic.json",
        help="path to intrinsic.json file",
    )
    parser.add_argument(
        "--source",
        "-s",
        default="basler",
        help="camera source directory or camer type",
    )

    args = parser.parse_args()
    if os.path.exists(args.intrinsic):
        intrinsic = Intrinsic.load(args.intrinsic)
    else:
        intrinsic = None

    if os.path.exists(args.source):
        camera = ImageLoader(args.source)
    elif args.source == "basler":
        camera = Basler(model="acA5472-5gc",intrinsic=None)
    # cam = Basler(model='acA2500-60uc')
    # print(cam.intrinsic.camera_matrix)
    # print(cam.intrinsic.distortion)

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    dd = datetime.strftime(datetime.now(), "%m%d")
    i = 0
    win_name = "color"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    while True:
        # get rgb-d from azure kinect
        bgr = camera.get()
        cv2.imshow(win_name, bgr)
        k = cv2.waitKey(0)
        if k == ord("q"):
            break

        elif k == ord("s"):
            pre_fix = os.path.join(args.out_dir, f"{dd}_{i:04}")
            cv2.imwrite(f"{pre_fix}-color.png", bgr)
            print("Save ", pre_fix)

            i += 1
