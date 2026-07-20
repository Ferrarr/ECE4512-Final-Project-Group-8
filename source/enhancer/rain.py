import numpy as np

def detect_rain_angle(gray, low_r=10, angle_bins=180, exclude_axes=True):
    h, w = gray.shape
    win = np.outer(np.hanning(h), np.hanning(w))
    f = np.fft.fftshift(np.fft.fft2(gray.astype(np.float32) * win))
    mag_log = np.log1p(np.abs(f))

    cy, cx = h // 2, w // 2
    high_r = min(cy, cx) - 5

    y, x = np.indices((h, w))
    r = np.hypot(y - cy, x - cx)
    theta = np.degrees(np.arctan2(y - cy, x - cx)) % 180

    ring = (r >= low_r) & (r <= high_r)

    edges = np.linspace(0, 180, angle_bins + 1)
    idx = np.clip(np.digitize(theta[ring], edges) - 1, 0, angle_bins - 1)
    energy = np.zeros(angle_bins); counts = np.zeros(angle_bins)
    np.add.at(energy, idx, mag_log[ring])
    np.add.at(counts, idx, 1)
    profile = energy / np.maximum(counts, 1)

    if exclude_axes:
        search = profile.copy()
        for a in (0, 90, 180):
            band = np.minimum(np.abs(edges[:-1] - a), 180 - np.abs(edges[:-1] - a)) < 3
            search[band] *= 0.4
    else:
        search = profile

    peak = np.argmax(search)
    angle = (edges[peak] + edges[peak + 1]) / 2
    return angle, profile, edges


def build_mask(shape, angle_deg, angle_width=3, low_r=8):
    h, w = shape
    cy, cx = h // 2, w // 2
    high_r = min(cy, cx) - 5
    y, x = np.indices((h, w))
    r = np.hypot(y - cy, x - cx)
    theta = np.degrees(np.arctan2(y - cy, x - cx)) % 180

    d = np.minimum(np.abs(theta - angle_deg), 180 - np.abs(theta - angle_deg))
    atten = np.exp(-0.5 * (d / max(angle_width, 1e-3)) ** 2)
    mask = 1.0 - atten
    keep_ring = (r < low_r) | (r > high_r)
    mask[keep_ring] = 1.0
    return mask.astype(np.float32)


def remove_rain_fft(gray, angle_width=3, low_r=8):
    h, w = gray.shape
    angle, _, _ = detect_rain_angle(gray, low_r=low_r)
    f = np.fft.fftshift(np.fft.fft2(gray.astype(np.float32)))
    mask = build_mask((h, w), angle, angle_width, low_r)
    f_filt = f * mask
    out = np.abs(np.fft.ifft2(np.fft.ifftshift(f_filt)))
    return np.clip(out, 0, 255).astype(np.uint8)
