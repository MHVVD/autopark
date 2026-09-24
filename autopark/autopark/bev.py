"""Bird's-eye view: inverse perspective mapping of the rig cameras onto the ground plane.

No ROS imports. Coordinates are in the `ground` frame of autopark_sim.rig (below base_link,
z up, same yaw), so a BEV pixel is a metric position around the car.

BEV image layout: top = forward (+x), left = vehicle left (+y). Pixel (row r, col c) centre:
    x = x_max - (r + 0.5) * res,   y = y_max - (c + 0.5) * res
"""
from dataclasses import dataclass

import cv2
import numpy as np

from autopark_sim.rig import BODY_BOX, CAMERAS, FOOTPRINT, T_ground_cam


@dataclass(frozen=True)
class BevGrid:
    x_min: float = -7.65   # m, centred on the car (base_link x = 1.35 is the car centre)
    x_max: float = 10.35
    y_min: float = -9.0
    y_max: float = 9.0
    res: float = 0.04      # m per pixel

    @property
    def shape(self):
        return (int(round((self.x_max - self.x_min) / self.res)),
                int(round((self.y_max - self.y_min) / self.res)))

    def pixel_centres(self):
        """(H, W) arrays of the ground x, y of every BEV pixel centre."""
        h, w = self.shape
        r, c = np.mgrid[0:h, 0:w]
        return self.x_max - (r + 0.5) * self.res, self.y_max - (c + 0.5) * self.res

    def to_pixel(self, x, y):
        """Ground (x, y) -> fractional (col, row) in the BEV image (OpenCV point order)."""
        x, y = np.asarray(x, float), np.asarray(y, float)
        return (self.y_max - y) / self.res - 0.5, (self.x_max - x) / self.res - 0.5

    def to_ground(self, col, row):
        col, row = np.asarray(col, float), np.asarray(row, float)
        return self.x_max - (row + 0.5) * self.res, self.y_max - (col + 0.5) * self.res


class CameraModel:
    """Webots camera (axes: x optical, y left, z up; pixel u right, v down, index coordinates)
    posed in the ground frame. Supports planar (pinhole) and spherical (equidistant fisheye)."""

    def __init__(self, spec, T_ground_cam_=None):
        self.spec = spec
        self.K = spec.K()
        T = T_ground_cam(spec) if T_ground_cam_ is None else T_ground_cam_
        self.R = T[:3, :3]          # camera -> ground rotation
        self.t = T[:3, 3]           # camera position in ground

    def dir_to_pixel(self, d):
        """Camera-frame directions (N, 3) -> u, v (N,) and a mask of directions that are imaged."""
        s = self.spec
        x, y, z = d[:, 0], d[:, 1], d[:, 2]
        with np.errstate(divide='ignore', invalid='ignore'):
            if s.projection == 'planar':
                u = s.cx - self.K[0, 0] * y / x
                v = s.cy - self.K[1, 1] * z / x
                front = x > 1e-6
            else:  # spherical: equidistant, radius in normalised texture units = theta / hfov
                rho = np.hypot(y, z)
                theta = np.arctan2(rho, x)
                r = theta / s.hfov
                cu = np.where(rho > 1e-12, -y / rho, 0.0)
                cv = np.where(rho > 1e-12, -z / rho, 0.0)
                u = s.cx + r * cu * s.width
                v = s.cy + r * cv * s.height
                front = np.ones_like(x, bool)
        ok = front & (u >= 0) & (u <= s.width - 1) & (v >= 0) & (v <= s.height - 1)
        return u, v, ok

    def pixel_to_dir(self, u, v):
        """Pixel (index coordinates) -> unit direction in the camera frame, shape (..., 3)."""
        s = self.spec
        u, v = np.asarray(u, float), np.asarray(v, float)
        if s.projection == 'planar':
            d = np.stack([np.ones_like(u), -(u - s.cx) / self.K[0, 0],
                          -(v - s.cy) / self.K[1, 1]], axis=-1)
            return d / np.linalg.norm(d, axis=-1, keepdims=True)
        dx, dy = (u - s.cx) / s.width, (v - s.cy) / s.height
        r = np.hypot(dx, dy)
        theta = r * s.hfov
        with np.errstate(divide='ignore', invalid='ignore'):
            cu = np.where(r > 1e-12, dx / r, 0.0)
            cv = np.where(r > 1e-12, dy / r, 0.0)
        return np.stack([np.cos(theta), -np.sin(theta) * cu, -np.sin(theta) * cv], axis=-1)

    def project(self, pts):
        """Ground-frame points (N, 3) -> pixel u, v (N,) and a mask of points inside the image
        (and more than 5 cm in front of the camera)."""
        pc = (np.asarray(pts, float) - self.t) @ self.R  # R^T (p - t), row-vector form
        u, v, ok = self.dir_to_pixel(pc)
        return u, v, ok & (np.linalg.norm(pc, axis=1) > 0.05)

    def ray_to_ground(self, u, v):
        """Pixel -> intersection of its ray with the ground plane z = 0 (NaN if the ray does
        not hit the ground)."""
        d = self.pixel_to_dir(u, v) @ self.R.T
        with np.errstate(divide='ignore', invalid='ignore'):
            s = -self.t[2] / d[..., 2]
        s = np.where(s > 0, s, np.nan)
        return self.t[0] + s * d[..., 0], self.t[1] + s * d[..., 1]


def segment_hits_box(p0, p1, box=BODY_BOX):
    """For segments p0 (3,) -> p1 (N, 3): True where the segment passes through the box
    (slab test on the open interval, so endpoints on the surface do not count)."""
    lo = np.array(box[0::2], float)
    hi = np.array(box[1::2], float)
    d = p1 - p0
    with np.errstate(divide='ignore', invalid='ignore'):
        t1 = (lo - p0) / d
        t2 = (hi - p0) / d
    tmin = np.where(np.isnan(t1), -np.inf, np.minimum(t1, t2))
    tmax = np.where(np.isnan(t1), np.inf, np.maximum(t1, t2))
    # axis-parallel segment component outside the slab never hits
    outside = (d == 0) & ((p0 < lo) | (p0 > hi))
    t_enter = np.max(tmin, axis=1)
    t_exit = np.min(tmax, axis=1)
    return (t_enter < t_exit) & (t_exit > 1e-6) & (t_enter < 1 - 1e-6) & ~outside.any(axis=1)


class BevStitcher:
    """Precomputed remap tables: BEV pixel -> camera pixel for every camera, with blend weights.

    Weight of a camera at a ground point = (sin of the viewing elevation)^2, i.e. cameras that
    look more steeply down (higher ground resolution, less stretching) dominate, feathered to 0
    towards the image border. Ground points hidden from a camera by the own car (BODY_BOX) or
    covered by its body mask get weight 0. The car footprint is black.
    """

    def __init__(self, grid=BevGrid(), cameras=CAMERAS, border=0.06, masks=None):
        """masks: optional dict camera name -> (H, W) bool image, True where the camera sees
        its own car body (never projected onto the ground)."""
        self.grid = grid
        self.names = [c.name for c in cameras]
        masks = masks or {}
        gx, gy = grid.pixel_centres()
        pts = np.stack([gx.ravel(), gy.ravel(), np.zeros(gx.size)], axis=1)
        x0, x1, y0, y1 = FOOTPRINT
        on_car = (gx >= x0) & (gx <= x1) & (gy >= y0) & (gy <= y1)

        self.maps, weights = [], []
        for spec in cameras:
            cam = CameraModel(spec)
            u, v, ok = cam.project(pts)
            ok &= ~segment_hits_box(cam.t, pts)      # line of sight blocked by the own car
            rng = np.linalg.norm(pts - cam.t, axis=1)
            elev = cam.t[2] / np.maximum(rng, 1e-6)          # sin(elevation)
            # feather: distance to the image border as a fraction of the image size
            fu = np.minimum(u, spec.width - 1 - u) / spec.width
            fv = np.minimum(v, spec.height - 1 - v) / spec.height
            feather = np.clip(np.minimum(fu, fv) / border, 0, 1)
            if spec.name in masks:
                body = masks[spec.name]
                ui = np.clip(np.round(np.where(ok, u, 0)).astype(int), 0, spec.width - 1)
                vi = np.clip(np.round(np.where(ok, v, 0)).astype(int), 0, spec.height - 1)
                ok = ok & ~body[vi, ui]
            w = np.where(ok, elev ** 2 * feather, 0.0).reshape(gx.shape)
            w[on_car] = 0.0
            self.maps.append((np.where(ok, u, -1).reshape(gx.shape).astype(np.float32),
                              np.where(ok, v, -1).reshape(gx.shape).astype(np.float32)))
            weights.append(w)
        wsum = np.sum(weights, axis=0)
        self.valid = wsum > 0
        self.weights = [(w / np.where(self.valid, wsum, 1.0))[..., None].astype(np.float32)
                        for w in weights]

    def stitch(self, images):
        """images: dict name -> HxWx3 uint8 (missing cameras are skipped). Returns BEV uint8."""
        h, w = self.grid.shape
        out = np.zeros((h, w, 3), np.float32)
        for name, (mx, my), wt in zip(self.names, self.maps, self.weights):
            img = images.get(name)
            if img is None:
                continue
            warped = cv2.remap(img, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
            out += warped.astype(np.float32) * wt
        return np.clip(out, 0, 255).astype(np.uint8)


def load_body_masks(directory, cameras=CAMERAS):
    """Read body_<name>.png masks (255 = own car body) written by the calibrate_masks tool."""
    import os
    masks = {}
    for spec in cameras:
        path = os.path.join(directory, f'body_{spec.name}.png')
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(path)
        if img.shape != (spec.height, spec.width):
            raise ValueError(f'{path}: shape {img.shape}, expected {(spec.height, spec.width)}')
        masks[spec.name] = img > 127
    return masks
