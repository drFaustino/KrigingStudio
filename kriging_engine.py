# -*- coding: utf-8 -*-
"""Robust point-kriging engine used by Kriging Studio.

The formulation follows the standard geostatistical point-kriging system used
by Surfer: Ordinary Kriging is the no-drift case; Universal Kriging uses
linear or quadratic drift terms. The implementation is independent and does
not claim to reproduce proprietary Surfer internals bit-for-bit.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.distance import cdist


class KrigingEngine:
    """Numerically robust Ordinary/Universal Point Kriging engine."""

    SUPPORTED_MODELS = ("linear", "spherical", "exponential", "gaussian",
                        "power", "hole_effect")

    def __init__(self):
        self.model_type = "linear"
        self.nugget = 0.0
        self.sill = 1.0
        self.range_val = 100.0
        self.slope = 1.0
        self.exponent = 1.0
        self.scale = 1.0
        self.anis_ratio = 1.0
        self.anis_angle = 0.0
        self.drift_type = "None"
        self.jitter = 1e-10
        self.max_neighbors = None
        self.min_neighbors = 3
        self.last_error = ""
        self._drift_center = np.zeros(2, dtype=float)
        self._drift_scale = np.ones(2, dtype=float)
        self.progress_callback = None

    @staticmethod
    def _as_points(a, name):
        a = np.asarray(a, dtype=float)
        if a.ndim != 2 or a.shape[1] != 2:
            raise ValueError(f"{name} must be an N x 2 array.")
        if not np.all(np.isfinite(a)):
            raise ValueError(f"{name} contains non-finite coordinates.")
        return a

    @staticmethod
    def _deduplicate(coords, values):
        """Average exact coordinate duplicates for a stable kriging system."""
        if len(coords) == 0:
            return coords, values
        unique, inverse = np.unique(coords, axis=0, return_inverse=True)
        if len(unique) == len(coords):
            return coords, values
        sums = np.bincount(inverse, weights=values)
        counts = np.bincount(inverse)
        return unique, sums / np.maximum(counts, 1)

    @staticmethod
    def _as_values(a, n):
        a = np.asarray(a, dtype=float).reshape(-1)
        if len(a) != n:
            raise ValueError("The number of values must match the number of points.")
        if not np.all(np.isfinite(a)):
            raise ValueError("The input values contain non-finite values.")
        return a

    def variogram_function(self, h, include_nugget=True):
        """Return semivariance for distance h.

        The nugget is applied only for h > 0. Therefore gamma(0)=0, which is
        required by the kriging matrix even when a nugget effect is present.
        """
        h = np.maximum(np.asarray(h, dtype=float), 0.0)
        positive = h > 0.0
        nug = self.nugget if include_nugget else 0.0
        r = max(float(self.range_val), np.finfo(float).eps)
        sill = max(float(self.sill), 0.0)

        model = str(self.model_type).lower()
        if model == "linear":
            gamma = self.slope * h
        elif model == "spherical":
            q = h / r
            gamma = np.where(q <= 1.0,
                             sill * (1.5 * q - 0.5 * q**3),
                             sill)
        elif model == "exponential":
            gamma = sill * (1.0 - np.exp(-h / r))
        elif model == "gaussian":
            gamma = sill * (1.0 - np.exp(-(h / r) ** 2))
        elif model == "power":
            gamma = max(float(self.scale), 0.0) * np.power(h, np.clip(self.exponent, 0.01, 2.0))
        elif model == "hole_effect":
            q = h / r
            gamma = sill * (1.0 - np.sin(q) / np.where(q == 0.0, 1.0, q))
        else:
            raise ValueError(f"Unsupported variogram model: {self.model_type}")

        gamma = np.asarray(gamma, dtype=float)
        if include_nugget:
            gamma = gamma + np.where(positive, nug, 0.0)
        gamma = np.maximum(gamma, 0.0)
        if gamma.ndim:
            gamma = gamma.copy()
            gamma[h == 0.0] = 0.0
        elif h == 0:
            gamma = 0.0
        return gamma

    def transform_coordinates(self, coords):
        """Apply the Surfer-style anisotropy convention to coordinates."""
        pts = self._as_points(coords, "Coordinates").copy()
        ratio = max(float(self.anis_ratio), np.finfo(float).eps)
        angle = np.radians(float(self.anis_angle))
        c, s = np.cos(angle), np.sin(angle)
        # Rotate into the principal direction, then scale the minor axis.
        x = pts[:, 0] * c + pts[:, 1] * s
        y = -pts[:, 0] * s + pts[:, 1] * c
        return np.column_stack((x, y / ratio))

    def calculate_experimental_variogram(
        self, coords, values, n_lags=15, max_dist=None, lag_width=None,
        direction=0.0, tolerance=90.0
    ):
        """Calculate the classical method-of-moments experimental variogram."""
        coords = self._as_points(coords, "Coordinates")
        values = self._as_values(values, len(coords))
        if len(coords) < 2:
            return np.array([]), np.array([]), np.array([])

        dx = coords[:, 0][:, None] - coords[:, 0][None, :]
        dy = coords[:, 1][:, None] - coords[:, 1][None, :]
        d = np.hypot(dx, dy)
        i, j = np.triu_indices(len(coords), k=1)
        dist = d[i, j]
        semivar = 0.5 * (values[i] - values[j]) ** 2

        tol = float(np.clip(tolerance, 0.0, 90.0))
        if tol < 90.0:
            az = np.degrees(np.arctan2(coords[j, 1] - coords[i, 1],
                                       coords[j, 0] - coords[i, 0])) % 180.0
            target = float(direction) % 180.0
            diff = np.abs((az - target + 90.0) % 180.0 - 90.0)
            keep = diff <= tol
            dist, semivar = dist[keep], semivar[keep]

        if len(dist) == 0:
            return np.array([]), np.array([]), np.array([])

        if max_dist is None or max_dist <= 0:
            max_dist = float(np.max(dist))
        max_dist = float(max_dist)
        n_lags = int(max(1, n_lags))
        if lag_width is None or lag_width <= 0:
            lag_width = max_dist / n_lags
        lag_width = float(lag_width)
        if lag_width <= 0:
            return np.array([]), np.array([]), np.array([])

        centers, gamma, counts = [], [], []
        for k in range(n_lags):
            lo, hi = k * lag_width, (k + 1) * lag_width
            mask = (dist >= lo) & (dist < hi)
            if k == n_lags - 1:
                mask |= (dist >= lo) & (dist <= max_dist)
            if np.any(mask):
                centers.append(lo + 0.5 * lag_width)
                gamma.append(float(np.mean(semivar[mask])))
                counts.append(int(np.sum(mask)))
        return np.asarray(centers), np.asarray(gamma), np.asarray(counts)

    def autofit(self, lags, gamma, weights=None):
        """Fit variogram parameters using weighted bounded least squares."""
        lags = np.asarray(lags, dtype=float).reshape(-1)
        gamma = np.asarray(gamma, dtype=float).reshape(-1)
        mask = np.isfinite(lags) & np.isfinite(gamma) & (lags > 0)
        lags, gamma = lags[mask], gamma[mask]
        if len(lags) < 2:
            return False
        if weights is None:
            weights = np.ones_like(lags)
        weights = np.maximum(np.asarray(weights, dtype=float), 1.0)

        model = self.model_type.lower()
        if model == "linear":
            x0 = [max(self.nugget, 0.0), max(self.slope, 1e-12)]
            def fun(x):
                self.nugget, self.slope = x
                return (self.variogram_function(lags) - gamma) / np.sqrt(weights)
            res = least_squares(fun, x0, bounds=([0.0, 1e-12], [np.inf, np.inf]), max_nfev=500)
            self.nugget, self.slope = map(float, res.x)
        elif model == "power":
            x0 = [max(self.scale, 1e-12), np.clip(self.exponent, 0.05, 1.99)]
            def fun(x):
                self.scale, self.exponent = x
                return (self.variogram_function(lags) - gamma) / np.sqrt(weights)
            res = least_squares(fun, x0, bounds=([1e-12, 0.01], [np.inf, 1.99]), max_nfev=500)
            self.scale, self.exponent = map(float, res.x)
        else:
            sill0 = max(float(np.nanmax(gamma)), 1e-12)
            range0 = max(float(np.nanmedian(lags)), 1e-12)
            x0 = [max(self.sill, sill0), max(self.range_val, range0)]
            def fun(x):
                self.sill, self.range_val = x
                return (self.variogram_function(lags) - gamma) / np.sqrt(weights)
            res = least_squares(fun, x0, bounds=([1e-12, 1e-12], [np.inf, np.inf]), max_nfev=500)
            self.sill, self.range_val = map(float, res.x)
        return bool(res.success)

    def build_drift_matrix(self, coords, set_reference=False):
        coords = self._as_points(coords, "Coordinates")
        x, y = coords[:, 0], coords[:, 1]
        # Scale coordinates for numerical conditioning while preserving the
        # same polynomial drift space.
        if set_reference:
            self._drift_center = np.array([np.mean(x) if len(x) else 0.0,
                                           np.mean(y) if len(y) else 0.0])
            self._drift_scale = np.array([max(np.ptp(x), 1.0) if len(x) else 1.0,
                                          max(np.ptp(y), 1.0) if len(y) else 1.0])
        xc = (x - self._drift_center[0]) / self._drift_scale[0]
        yc = (y - self._drift_center[1]) / self._drift_scale[1]
        if self.drift_type == "Linear":
            return np.column_stack((np.ones(len(coords)), xc, yc))
        if self.drift_type == "Quadratic":
            return np.column_stack((np.ones(len(coords)), xc, yc, xc**2, yc**2, xc*yc))
        return np.ones((len(coords), 1))

    @staticmethod
    def _stable_solve(a, b):
        try:
            return np.linalg.solve(a, b)
        except np.linalg.LinAlgError:
            # Symmetric systems can still be solved robustly with a small
            # diagonal regularisation. Fall back to least squares as a last resort.
            scale = max(float(np.nanmax(np.abs(a))), 1.0)
            eye = np.eye(a.shape[0]) * (1e-10 * scale)
            try:
                return np.linalg.solve(a + eye, b)
            except np.linalg.LinAlgError:
                return np.linalg.lstsq(a, b, rcond=1e-12)[0]

    def _select_neighbors(self, source, target, max_neighbors=None):
        if not max_neighbors or max_neighbors >= len(source):
            return np.arange(len(source))
        d = np.linalg.norm(source - target, axis=1)
        return np.argsort(d)[:max(3, int(max_neighbors))]

    def fit_and_predict(
        self, src_coords, src_values, grid_coords, grid_y=None,
        return_variance=True, max_neighbors=None, progress_callback=None
    ):
        """Predict point values.

        Backward-compatible API:
          fit_and_predict(coords, values, grid_coords)
          fit_and_predict(coords, values, grid_x, grid_y)

        The second form accepts meshgrid arrays and returns reshaped outputs.
        """
        src = self._as_points(src_coords, "Source coordinates")
        vals = self._as_values(src_values, len(src))
        src, vals = self._deduplicate(src, vals)
        if len(src) < 3:
            raise ValueError("At least three valid input points are required.")

        reshape = None
        if grid_y is not None:
            gx, gy = np.asarray(grid_coords, dtype=float), np.asarray(grid_y, dtype=float)
            if gx.shape != gy.shape:
                raise ValueError("Grid X and Grid Y must have the same shape.")
            reshape = gx.shape
            targets = np.column_stack((gx.ravel(), gy.ravel()))
        else:
            targets = self._as_points(grid_coords, "Grid coordinates")

        callback = progress_callback or self.progress_callback
        if callback:
            callback(0)
        src_t = self.transform_coordinates(src)
        target_t = self.transform_coordinates(targets)
        n = len(src)
        p = self.build_drift_matrix(src).shape[1]

        # Global system is reused when no local neighbourhood is requested.
        local = max_neighbors if max_neighbors is not None else self.max_neighbors
        use_global = not local or int(local) >= n
        if use_global:
            F = self.build_drift_matrix(src, set_reference=True)
            D = cdist(src_t, src_t)
            Gamma = self.variogram_function(D)
            np.fill_diagonal(Gamma, 0.0)
            A = np.block([[Gamma, F], [F.T, np.zeros((p, p))]])
            out = np.empty(len(targets), dtype=float)
            var = np.full(len(targets), np.nan, dtype=float)
            for start in range(0, len(targets), 4096):
                stop = min(start + 4096, len(targets))
                Dt = cdist(src_t, target_t[start:stop])
                g = self.variogram_function(Dt)
                Ft = self.build_drift_matrix(targets[start:stop])
                B = np.vstack((g, Ft.T))
                sol = self._stable_solve(A, B)
                out[start:stop] = vals @ sol[:n]
                if return_variance:
                    var[start:stop] = np.sum(sol * B, axis=0)
                if callback:
                    callback(int(100 * stop / max(len(targets), 1)))
        else:
            out = np.empty(len(targets), dtype=float)
            var = np.full(len(targets), np.nan, dtype=float)
            for k, target in enumerate(targets):
                idx = self._select_neighbors(src_t, target_t[k], local)
                s = src[idx]
                st = src_t[idx]
                v = vals[idx]
                F = self.build_drift_matrix(s, set_reference=True)
                pp = F.shape[1]
                Gamma = self.variogram_function(cdist(st, st))
                np.fill_diagonal(Gamma, 0.0)
                A = np.block([[Gamma, F], [F.T, np.zeros((pp, pp))]])
                g = self.variogram_function(cdist(st, target_t[k:k+1]).ravel())
                Ft = self.build_drift_matrix(target.reshape(1, 2))
                B = np.concatenate((g, Ft.ravel()))
                sol = self._stable_solve(A, B)
                out[k] = float(v @ sol[:len(idx)])
                if return_variance:
                    var[k] = float(sol @ B)
                if callback and (k == len(targets) - 1 or k % max(1, len(targets)//100) == 0):
                    callback(int(100 * (k + 1) / max(len(targets), 1)))

        if callback:
            callback(100)
        if reshape is not None:
            return out.reshape(reshape), var.reshape(reshape)
        return out, var

    def cross_validate(self, coords, values, max_neighbors=None, progress_callback=None):
        """Leave-one-out cross-validation using the current kriging model."""
        coords = self._as_points(coords, "Coordinates")
        values = self._as_values(values, len(coords))
        if len(coords) < 4:
            raise ValueError("At least four points are required for cross-validation.")
        estimates = np.full(len(values), np.nan)
        callback = progress_callback or self.progress_callback
        if callback:
            callback(0)
        for i in range(len(values)):
            mask = np.ones(len(values), dtype=bool)
            mask[i] = False
            pred, _ = self.fit_and_predict(
                coords[mask], values[mask], coords[i:i+1],
                return_variance=False, max_neighbors=max_neighbors
            )
            estimates[i] = pred[0]
            if callback:
                callback(int(100 * (i + 1) / len(values)))
        errors = values - estimates
        if callback:
            callback(100)
        return values, estimates, errors
