import logging

import numpy as np
from scipy.signal import find_peaks
from skimage.feature import peak_local_max

__all__ = ["PSFAverage"]

logger = logging.getLogger(__name__)


class PSFAverage(object):
    """
    Select peaks over specified detection ratio and extract their PSF info.

    Args:
        ratio (float): ratio of S.D. criterion
    """

    def __init__(self, bbox, ratio=2):
        self._bbox = self._normalize_bbox(bbox)
        self._ratio = ratio

    def __call__(self, data, return_coords=False):
        if not (data.ndim == 2 or data.ndim == 3):
            raise ValueError("input has to be 2-D or 3-D")

        if data.ndim == 2:
            coords = self._select_candidates(data)
        else:
            coords = self._select_candidates(np.max(data, axis=0))
        peaks = self._crop_candidates(data, coords, return_coords)

        #TODO fit each 3d psf

        return peaks

    ##

    @property
    def bbox(self):
        return self._bbox

    @property
    def ratio(self):
        return self._ratio

    ##

    def _normalize_bbox(self, bbox):
        """Ensure bounding box is odd number in all dimension."""
        new_bbox = tuple(dim // 2 * 2 + 1 for dim in bbox)
        if any(new != old for new, old in zip(new_bbox, bbox)):
            logger.warning(f"resize bounding box from {bbox} to {new_bbox}")
        return new_bbox

    def _select_candidates(self, data_mip):
        threshold = data_mip.mean() + data_mip.std() * self.ratio
        _, dist1, dist2 = self.bbox
        # radius
        dist1, dist2 = dist1 // 2, dist2 // 2
        coords = peak_local_max(
            data_mip, min_distance=max(dist1, dist2), threshold_abs=threshold
        )
        logger.info(f"found {coords.shape[0]} peaks")
        return coords

    def _crop_candidates(self, data, coords, return_coords=False):
        results = []
        bz, by, bx = self.bbox
        rz, ry, rx = bz // 2, by // 2, bx // 2
        for cy, cx in coords:
            # extract along lateral
            roi_slice = (slice(cy - ry, cy + ry + 1), slice(cx - rx, cx + rx + 1))
            if data.ndim == 3:
                roi_slice = (slice(None),) + roi_slice
            roi = data[roi_slice]

            # extract along axial
            if data.ndim == 3:
                # flatten
                shape = roi.shape
                roi = np.reshape(roi, (shape[0], -1))
                # find max along axial direction
                z = roi.sum(axis=1)
                threshold = z.mean()
                peaks, properties = find_peaks(z)
                if len(peaks) == 1:
                    cz = peaks[0]
                elif len(peaks) > 1:
                    cz = peaks[np.argmax(z[peaks])]
                else:
                    logger.debug(f".. ({cx}, {cy}, {cz}), ambiguous z profile")
                    continue
                # unflatten
                roi = np.reshape(roi, shape)

                if not (rz <= cz < shape[0] - rz):
                    logger.debug(f".. ({cx}, {cy}, {cz}), insufficient z height")
                    continue
                result = roi[cz - rz : cz + rz + 1, ...]
                if return_coords:
                    result = (result, (cz, cy, cx))
            else:
                result = roi
                if return_coords:
                    result = (result, (cy, cx))
            results.append(result)

        logger.info(f"{len(results)} peaks cropped")
        return results

