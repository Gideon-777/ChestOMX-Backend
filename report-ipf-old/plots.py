import numpy as np
import SimpleITK as sitk
import k3d
import tqdm
from widgets import *


def get_poly(masks, prefix, size_reduce=[1, 1, 1], smooth=18, names=None):
    poly = {}
    img = sitk.GetArrayFromImage(masks).astype(np.float32)
    bounds = get_bounds(masks)
    volumes = {}

    for i in np.unique(img)[1:]:
        center = np.median(np.dstack(np.where(img == i)), axis=1)[0] // np.array(size_reduce)

        if names is None or str(int(i)) not in names.keys():
            name = prefix + ' ' + str(int(i))
        else:
            name = names[str(int(i))]

        volumes[str(i)] = {
            'Label': name,
            'name': name.replace(' ', '_'),
            'id': [int(i)],
            'center': center[::-1].astype(np.int32).tolist(),
            'Volume': np.sum(img == i) * np.prod(np.array(masks.GetSpacing())) / (10 ** 3)
        }

    for idx, val in tqdm.tqdm(list(enumerate(np.unique(img)[1:]))):
        p = contour(img == val, bounds, np.array([0.5]), 0.0, smooth)

        name = prefix + '_' + str(int(val))
        poly[name] = k3d.vtk_poly_data(p, name=name, opacity=1.0, color=k3d.nice_colors[idx + 1])

    return poly, img.astype(np.uint8), volumes
