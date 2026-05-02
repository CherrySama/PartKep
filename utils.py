"""
Created by Yinghao Ho on 2026-1-24

Coordinate transformation utilities.
Covers 2D bbox transforms, depth unprojection, and SE3 helpers (Rodrigues, matrix4x4).
"""

import sys
from typing import Tuple, List, Union, Optional, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from configs.camera_config import CameraConfig


class CoordinateTransformer:
    """
    Static utility class for coordinate transforms used across the pipeline.

    2D: normalised <-> absolute bbox, ROI <-> original image, bilinear depth sampling.
    3D: pixel + depth -> camera -> world, sim mode passthrough.
    SE3: axis-angle (Rodrigues) -> rotation matrix, position + R -> 4x4 homogeneous T.
    """

    @staticmethod
    def normalize_to_absolute(bbox_norm: Union[List[float], np.ndarray],
                              img_w: int,
                              img_h: int) -> Tuple[float, float, float, float]:
        """[x1,y1,x2,y2] normalised [0,1] -> absolute float pixel coords."""
        if len(bbox_norm) != 4:
            raise ValueError(f"bbox_norm must have 4 elements, got {len(bbox_norm)}")
        x1_float = float(bbox_norm[0]) * img_w
        y1_float = float(bbox_norm[1]) * img_h
        x2_float = float(bbox_norm[2]) * img_w
        y2_float = float(bbox_norm[3]) * img_h
        return x1_float, y1_float, x2_float, y2_float

    @staticmethod
    def absolute_to_normalize(bbox_abs: Union[List[float], np.ndarray],
                              img_w: int,
                              img_h: int) -> Tuple[float, float, float, float]:
        """[x1,y1,x2,y2] absolute float pixel coords -> normalised [0,1]."""
        if len(bbox_abs) != 4:
            raise ValueError(f"bbox_abs must have 4 elements, got {len(bbox_abs)}")
        if img_w <= 0 or img_h <= 0:
            raise ValueError(f"Image dimensions must be positive, got width={img_w}, height={img_h}")
        x1_norm = float(bbox_abs[0]) / img_w
        y1_norm = float(bbox_abs[1]) / img_h
        x2_norm = float(bbox_abs[2]) / img_w
        y2_norm = float(bbox_abs[3]) / img_h
        return x1_norm, y1_norm, x2_norm, y2_norm

    @staticmethod
    def get_crop_bbox(x1_float: float, y1_float: float,
                      x2_float: float, y2_float: float,
                      img_w: int, img_h: int) -> Tuple[int, int, int, int]:
        """Float bbox -> integer crop coords, clamped to image bounds."""
        if img_w <= 0 or img_h <= 0:
            raise ValueError(f"Image dimensions must be positive, got width={img_w}, height={img_h}")
        x1_crop = max(0, min(int(x1_float), img_w - 1))
        y1_crop = max(0, min(int(y1_float), img_h - 1))
        x2_crop = max(0, min(int(x2_float), img_w))
        y2_crop = max(0, min(int(y2_float), img_h))

        if x2_crop <= x1_crop:
            x2_crop = min(x1_crop + 1, img_w)
        if y2_crop <= y1_crop:
            y2_crop = min(y1_crop + 1, img_h)

        return x1_crop, y1_crop, x2_crop, y2_crop

    @staticmethod
    def transform_point_roi_to_original(
            point_roi: Tuple[float, float],
            roi_offset_x: int,
            roi_offset_y: int) -> Tuple[float, float]:
        """Single point: ROI coords -> original image coords."""
        x_roi, y_roi = point_roi
        return float(x_roi) + roi_offset_x, float(y_roi) + roi_offset_y

    @staticmethod
    def transform_points_batch(points_roi: np.ndarray,
                               roi_offset_x: int,
                               roi_offset_y: int) -> np.ndarray:
        """Batch points (N,2): ROI coords -> original image coords."""
        if not isinstance(points_roi, np.ndarray):
            points_roi = np.array(points_roi)
        if points_roi.ndim != 2 or points_roi.shape[1] != 2:
            raise ValueError(f"points_roi must be (N,2), got {points_roi.shape}")
        points_original = points_roi.copy().astype(np.float64)
        points_original[:, 0] += roi_offset_x
        points_original[:, 1] += roi_offset_y
        return points_original

    @staticmethod
    def get_depth_bilinear(depth_map: np.ndarray, x: float, y: float) -> float:
        """Bilinear interpolation of depth at float pixel (x, y). Returns 0.0 if out of bounds."""
        h, w = depth_map.shape
        if x < 0 or x >= w or y < 0 or y >= h:
            return 0.0

        x0, x1 = int(np.floor(x)), min(int(np.floor(x)) + 1, w - 1)
        y0, y1 = int(np.floor(y)), min(int(np.floor(y)) + 1, h - 1)
        dx, dy = x - x0, y - y0

        d00 = float(depth_map[y0, x0])
        d01 = float(depth_map[y0, x1])
        d10 = float(depth_map[y1, x0])
        d11 = float(depth_map[y1, x1])

        return (d00 * (1 - dx) * (1 - dy) +
                d01 * dx * (1 - dy) +
                d10 * (1 - dx) * dy +
                d11 * dx * dy)

    @staticmethod
    def verify_transform(bbox_norm: List[float],
                         img_w: int,
                         img_h: int,
                         point_roi: Tuple[float, float],
                         point_original: Tuple[float, float],
                         eps: float = 1e-6) -> bool:
        """Debug helper: verify ROI->original transform is consistent with bbox_norm."""
        x1_f, y1_f, x2_f, y2_f = CoordinateTransformer.normalize_to_absolute(
            bbox_norm, img_w, img_h
        )
        x1_c, y1_c, x2_c, y2_c = CoordinateTransformer.get_crop_bbox(
            x1_f, y1_f, x2_f, y2_f, img_w, img_h
        )
        x_calc, y_calc = CoordinateTransformer.transform_point_roi_to_original(
            point_roi, x1_c, y1_c
        )
        x_diff = abs(x_calc - point_original[0])
        y_diff = abs(y_calc - point_original[1])
        is_correct = (x_diff < eps and y_diff < eps)
        if not is_correct:
            print(f"[verify_transform] mismatch: expected {point_original}, "
                  f"got ({x_calc:.6f}, {y_calc:.6f})")
        return is_correct

    @staticmethod
    def get_roi_info(bbox_norm: List[float], img_w: int, img_h: int) -> dict:
        """Debug helper: returns bbox in all representations."""
        x1_f, y1_f, x2_f, y2_f = CoordinateTransformer.normalize_to_absolute(
            bbox_norm, img_w, img_h
        )
        x1_c, y1_c, x2_c, y2_c = CoordinateTransformer.get_crop_bbox(
            x1_f, y1_f, x2_f, y2_f, img_w, img_h
        )
        return {
            'bbox_norm':  bbox_norm,
            'bbox_float': (x1_f, y1_f, x2_f, y2_f),
            'bbox_crop':  (x1_c, y1_c, x2_c, y2_c),
            'roi_size':   (x2_c - x1_c, y2_c - y1_c),
            'offset':     (x1_c, y1_c)
        }

    @staticmethod
    def pixel_to_camera_3d(x_pixel: float,
                           y_pixel: float,
                           depth_m: float,
                           camera_config: 'CameraConfig') -> np.ndarray:
        """
        Pixel + depth -> 3D point in camera frame.
        Pinhole back-projection: X = (x - cx) * depth / fx, Y = (y - cy) * depth / fy, Z = depth.
        """
        if depth_m <= 0:
            raise ValueError(f"depth_m must be > 0, got {depth_m}")
        X_cam = (x_pixel - camera_config.cx) * depth_m / camera_config.fx
        Y_cam = (y_pixel - camera_config.cy) * depth_m / camera_config.fy
        return np.array([X_cam, Y_cam, float(depth_m)], dtype=np.float64)

    @staticmethod
    def camera_to_world_3d(point_camera: np.ndarray,
                           camera_config: 'CameraConfig') -> np.ndarray:
        """3D point in camera frame -> world frame via extrinsic matrix."""
        point_camera = np.array(point_camera, dtype=np.float64).flatten()
        if point_camera.shape != (3,):
            raise ValueError(f"point_camera must be shape (3,), got {point_camera.shape}")
        point_homogeneous = np.append(point_camera, 1.0)
        point_world_h = camera_config.extrinsic_matrix @ point_homogeneous
        return point_world_h[:3]

    @staticmethod
    def get_keypoint_3d(
            mode: str,
            camera_config: Optional['CameraConfig'] = None,
            x_pixel: Optional[float] = None,
            y_pixel: Optional[float] = None,
            depth_map: Optional[np.ndarray] = None,
            depth_m: Optional[float] = None,
            xyz_world: Optional[Union[np.ndarray, List[float]]] = None
    ) -> np.ndarray:
        """
        Unified 3D keypoint interface.

        mode='sim': passthrough MuJoCo world coords (xyz_world required).
        mode='real': pixel + depth -> camera -> world (camera_config, x_pixel, y_pixel required;
                     depth_map or depth_m required).
        Returns shape=(3,) world-frame 3D point (metres).
        """
        if mode == "sim":
            if xyz_world is None:
                raise ValueError("sim mode requires xyz_world")
            result = np.array(xyz_world, dtype=np.float64).flatten()
            if result.shape != (3,):
                raise ValueError(f"xyz_world must be shape (3,), got {result.shape}")
            return result

        elif mode == "real":
            if camera_config is None:
                raise ValueError("real mode requires camera_config")
            if x_pixel is None or y_pixel is None:
                raise ValueError("real mode requires x_pixel and y_pixel")
            if depth_map is None and depth_m is None:
                raise ValueError("real mode requires depth_map or depth_m")

            if depth_m is None:
                depth_mm = CoordinateTransformer.get_depth_bilinear(depth_map, x_pixel, y_pixel)
                if depth_mm <= 0:
                    raise ValueError(
                        f"invalid depth at pixel ({x_pixel:.1f}, {y_pixel:.1f}): {depth_mm}mm"
                    )
                depth_m = depth_mm / 1000.0

            point_camera = CoordinateTransformer.pixel_to_camera_3d(
                x_pixel, y_pixel, depth_m, camera_config
            )
            return CoordinateTransformer.camera_to_world_3d(point_camera, camera_config)

        else:
            raise ValueError(f"mode must be 'real' or 'sim', got '{mode}'")

    @staticmethod
    def rodrigues(rvec: np.ndarray) -> np.ndarray:
        """
        Axis-angle vector -> 3x3 rotation matrix (Rodrigues formula).
        R = I + sin(theta)*K + (1-cos(theta))*K^2, where K is the skew-symmetric matrix of k=rvec/theta.
        Zero vector -> identity.
        """
        rvec = np.array(rvec, dtype=np.float64)
        theta = np.linalg.norm(rvec)
        if theta < 1e-10:
            return np.eye(3)
        k = rvec / theta
        K = np.array([
            [  0.0, -k[2],  k[1]],
            [ k[2],   0.0, -k[0]],
            [-k[1],  k[0],   0.0]
        ])
        return np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)

    @staticmethod
    def rotation_to_matrix4x4(position: np.ndarray, R: np.ndarray) -> np.ndarray:
        """
        position (3,) + R (3,3) -> 4x4 homogeneous transform T = [[R, p], [0, 1]].
        Used by PoseSolver to produce T for IKSolver.
        """
        position = np.array(position, dtype=np.float64).flatten()
        R = np.array(R, dtype=np.float64)
        if position.shape != (3,):
            raise ValueError(f"position must be shape (3,), got {position.shape}")
        if R.shape != (3, 3):
            raise ValueError(f"R must be shape (3,3), got {R.shape}")
        T = np.eye(4)
        T[:3, :3] = R
        T[:3,  3] = position
        return T


if __name__ == "__main__":
    sys.path.insert(0, '.')
    from configs.camera_config import CameraConfig

    img_w, img_h = 640, 480
    bbox_norm = [0.234567, 0.456789, 0.567890, 0.789012]
    x1_f, y1_f, x2_f, y2_f = CoordinateTransformer.normalize_to_absolute(bbox_norm, img_w, img_h)
    x1_c, y1_c, x2_c, y2_c = CoordinateTransformer.get_crop_bbox(x1_f, y1_f, x2_f, y2_f, img_w, img_h)
    point_roi = (106.7823, 79.4512)
    x_orig, y_orig = CoordinateTransformer.transform_point_roi_to_original(point_roi, x1_c, y1_c)
    ok = CoordinateTransformer.verify_transform(bbox_norm, img_w, img_h, point_roi, (x_orig, y_orig))
    assert ok

    mujoco_xpos = np.array([0.156, 0.298, 0.045])
    p3d = CoordinateTransformer.get_keypoint_3d(mode="sim", xyz_world=mujoco_xpos)
    assert np.allclose(p3d, mujoco_xpos)

    R0 = CoordinateTransformer.rodrigues(np.zeros(3))
    assert np.allclose(R0, np.eye(3))

    R_90z = CoordinateTransformer.rodrigues(np.array([0.0, 0.0, np.pi / 2]))
    assert np.allclose(R_90z, np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=float), atol=1e-6)

    R_rand = CoordinateTransformer.rodrigues(np.array([0.3, -0.5, 0.8]))
    assert np.allclose(R_rand.T @ R_rand, np.eye(3), atol=1e-10)
    assert np.isclose(np.linalg.det(R_rand), 1.0, atol=1e-10)

    pos = np.array([0.5, 0.1, 0.3])
    R   = CoordinateTransformer.rodrigues(np.zeros(3))
    T   = CoordinateTransformer.rotation_to_matrix4x4(pos, R)
    assert T.shape == (4, 4)
    assert np.allclose(T[:3, 3], pos, atol=1e-10)
    assert np.allclose(T[:3, :3], R, atol=1e-10)
    assert np.allclose(T[3, :], [0, 0, 0, 1], atol=1e-10)

    print("all checks passed")