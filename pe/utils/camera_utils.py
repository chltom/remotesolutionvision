import json
import os
from dataclasses import dataclass, field
from dataclasses_json import DataClassJsonMixin, config, dataclass_json
from typing import Self, Tuple

import cv2
import numpy as np


def vector2Mat(rvec, tvec):
    mat = np.eye(4)
    mat[:3, :3] = cv2.Rodrigues(rvec)[0]
    mat[:3, 3] = tvec.squeeze()
    return mat


def rvect2Rmat(rvec):
    return cv2.Rodrigues(rvec)[0]


@dataclass
class Intrinsic(DataClassJsonMixin):
    width: int = 0
    height: int = 0
    fx: float = 0.0
    fy: float = 0.0
    cx: float = 0.0
    cy: float = 0.0
    camera_matrix: np.ndarray = field(
        default_factory=lambda: np.eye(3), metadata=config(decoder=np.asarray)
    )
    distortion: np.ndarray = field(
        default_factory=lambda: np.zeros(5), metadata=config(decoder=np.asarray)
    )

    def __post_init__(self):
        if isinstance(self.camera_matrix, list):
            self.camera_matrix = np.array(self.camera_matrix).reshape(3, 3)

        if self.cx + self.cy == 0:
            self.fx = self.camera_matrix[0, 0]
            self.fy = self.camera_matrix[1, 1]
            self.cx = self.camera_matrix[0, 2]
            self.cy = self.camera_matrix[1, 2]

        else:
            self.camera_matrix = np.array(
                [[self.fx, 0, self.cx], [0, self.fy, self.cy], [0, 0, 1.0]]
            )

        if isinstance(self.distortion, list):
            self.distortion = np.array(self.distortion).ravel()

        if self.distortion.ndim > 1:
            self.distortion = self.distortion.squeeze()

    def get_fxfycxcy(self) -> Tuple[float, float, float, float]:
        return (
            self.camera_matrix[0, 0],
            self.camera_matrix[1, 1],
            self.camera_matrix[0, 2],
            self.camera_matrix[1, 2],
        )

    @classmethod
    def load(cls, filename: str) -> Self:
        ext = os.path.splitext(filename)[-1].lower()
        assert ext == ".json", "Intrinsic file must be json!"
        try:
            with open(filename, "r", encoding="utf-8") as f:
                json_dict = json.load(f)
            return cls.from_dict(json_dict)
        except Exception as e:
            raise e

    def save(self, filename: str) -> None:
        try:
            with open(filename, "w", encoding="utf-8") as f:
                f.write(json.dumps(self.to_dict(), indent=4))
        except Exception as e:
            raise e


class ChessBoard:
    def __init__(self, length=0.024, size=(9, 6)):
        self.length = length
        self.size = size
        w, h = size
        objp = np.zeros((w * h, 3), np.float32)
        objp[:, :2] = np.mgrid[0:w, 0:h].T.reshape(-1, 2)
        self.objp = objp * length
        self.type = "chessboard"

        self.intrinsic = None
        # capture data
        self.chesspts = []
        self.imgpoints = []
        self.Rmats = []
        self.tvecs = []

    def detect(self, img, scale=3):
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        # criteria = (cv2.TERM_CRITERIA_EPS, 30, 0.001)
        flag = (
            cv2.CALIB_CB_ADAPTIVE_THRESH
            + cv2.CALIB_CB_NORMALIZE_IMAGE
            # + cv2.CALIB_CB_FAST_CHECK
        )
        # flag =  cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK
        if img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        scaled = cv2.resize(img, (img.shape[1] // scale, img.shape[0] // scale))
        self.ret, self.corners = cv2.findChessboardCorners(
            scaled, self.size, None, flags=flag
        )
        if self.ret:
            self.corners *= scale
            self.corners = cv2.cornerSubPix(
                img, self.corners, (11, 11), (-1, -1), criteria
            )
            # cv2.drawChessboardCorners(img, self.size, self.corners, self.ret)
            return True
        return self.ret

    def draw(self, img):
        img = cv2.drawChessboardCorners(img, self.size, self.corners, self.ret)
        return img

    def draw_axis(self, img):
        axis = np.float32(
            [[2 * self.length, 0, 0], [0, 2 * self.length, 0], [0, 0, 2 * self.length]]
        ).reshape(-1, 3)
        imgpts, jac = cv2.projectPoints(axis, self.rvecs, self.tvecs, self.cM, self.cD)
        imgpts = imgpts.astype(int)

        corner = tuple(self.corners[0].ravel().astype(int))
        img = cv2.line(img, corner, tuple(imgpts[0].ravel()), (0, 0, 255), 5)
        img = cv2.line(img, corner, tuple(imgpts[1].ravel()), (0, 255, 0), 5)
        img = cv2.line(img, corner, tuple(imgpts[2].ravel()), (255, 0, 0), 5)

    def set_intrinsic(self, intrinsic: Intrinsic):
        self.intrinsic = intrinsic

    def get_pose(self):
        if not self.ret:
            return [], [], []

        if self.intrinsic is not None:
            ret, self.rvec, self.tvec = cv2.solvePnP(
                self.objp, self.corners, self.cM, self.cD
            )
            return self.rvec.squeeze(), self.tvec.squeeze()
        else:
            return [], [], []

    def capture(self):
        self.chesspts.append(self.objp)
        self.imgpoints.append(self.corners)
        if self.intrinsic:
            rvec, tvec = self.get_pose()
            Rmat = rvect2Rmat(rvec)
            self.Rmats.append(Rmat)  # origin: cam
            self.tvecs.append(tvec)

    def get_captured_data(self):
        data = {
            "chesspts": self.chesspts,
            "imgpoints": self.imgpoints,
            "Rmats": self.Rmats,
            "tvecs": self.tvecs,
        }
        return data
