import SimpleITK as sitk
import k3d_pro as k3d
import numpy as np
import tqdm
from k3d_pro.headless import k3d_remote#, get_headless_driver

from widgets import *


def get_headless_driver(no_headless=False):
    from selenium.webdriver.chrome.options import Options
    from selenium import webdriver
    from webdriver_manager.chrome import ChromeDriverManager

    options = Options()

    if not no_headless:
        options.add_argument("--headless")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")

    return webdriver.Chrome(ChromeDriverManager().install(), options=options)


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


def get_screenshots(data, name="data", port=8080):
    ret = {}

    plot = k3d.plot(grid_visible=False, background_color=0,
                    camera_mode='slice_viewer', axes_helper=0,
                    screenshot_scale=3)

    plot.minimum_fps = -1
    headless = k3d_remote(plot, get_headless_driver(), 1024, 1024, port=port)
    headless.sync(hold_until_refreshed=True)

    for o in data.objects:
        if o['type'] == 'VolumeSlice':
            obj = o.clone()
            obj.compression_level = 0
            obj.color_range = [-1300, obj.color_range[1]]
            obj.active_masks = np.unique(obj.mask)[1:]
            plot += obj

    for axis in ['x', 'y', 'z']:
        plot.slice_viewer_direction = axis

        if len(plot.objects) > 0:
            plot.objects[0].slice_x = -1
            plot.objects[0].slice_y = -1
            plot.objects[0].slice_z = -1

            if axis == 'x':
                plot.objects[0].slice_x = plot.objects[0].volume.shape[2] // 4  # chest specific

            if axis == 'y':
                plot.objects[0].slice_y = plot.objects[0].volume.shape[1] // 2

            if axis == 'z':
                plot.objects[0].slice_z = plot.objects[0].volume.shape[0] // 2

        headless.sync(hold_until_refreshed=True)
        ret[name + "_" + axis] = headless.get_screenshot()

    while (len(plot.objects) > 0):
        plot -= plot.objects[-1]

    plot.camera_mode = 'trackball'
    headless.sync(hold_until_refreshed=True)

    for o in data.objects:
        if o.name.startswith("mesh_") or (name == 'Summary' and o['type'] == 'Mesh'):
            plot += o.clone()

    if len(plot.objects) > 0:
        plot.camera_auto_fit = False
        plot.colorbar_object_id = 0
        plot.camera = plot.get_auto_camera(0.8, -35, 85)
        headless.sync(hold_until_refreshed=True)
        ret[name + '_3d'] = headless.get_screenshot()

    headless.close()

    return ret
