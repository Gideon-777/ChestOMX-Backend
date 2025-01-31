import SimpleITK as sitk
import colorsys
import glob2
import k3d_pro as k3d
import numpy as np
import os
import pyacvd
import pyvista
import time
import vtk
from vtk.util import numpy_support

cm = []
for i in range(254):
    c = k3d.nice_colors[i % 19]
    cm += [(c >> (8 * (2 - i))) & 255 for i in range(3)]


def save_dicom(prefix, dicoms, folder_name, scan=None, mask=None, series_number=1):
    writer = sitk.ImageFileWriter()
    writer.KeepOriginalImageUIDOn()
    writer.UseCompressionOn()

    modification_time = time.strftime("%H%M%S")
    modification_date = time.strftime("%Y%m%d")

    if scan is not None and mask is not None:
        d = sitk.GetArrayFromImage(scan)

    window_minimum = np.percentile(d, 0)
    window_maximum = np.percentile(d, 95)

    for idx, image in enumerate(dicoms):
        if scan is not None and mask is not None:
            image_255 = sitk.Cast(sitk.IntensityWindowing(image,
                                                          windowMinimum=window_minimum,
                                                          windowMaximum=window_maximum,
                                                          outputMinimum=0.0,
                                                          outputMaximum=255.0), sitk.sitkUInt8)

            m = sitk.GetImageFromArray(np.expand_dims(mask[idx, :, :], axis=0))
            m.CopyInformation(image)

            final = sitk.LabelOverlay(image=image_255,
                                      labelImage=m,
                                      opacity=0.5, backgroundValue=0.0,
                                      colormap=cm)
        else:
            final = scan[idx]

        series_tag_values = [
            ("0008|0031", modification_time),  # Series Time
            ("0008|0012", modification_date),  # Instance Creation Date
            ("0008|0013", modification_time),  # Instance Creation Time
            ("0008|0021", modification_date),  # Series Date
            ("0008|103E", "[REPORT] " + prefix),  # Series Description
            ("0020|4000", "[REPORT]"),  # Image Comments
            ("0020|000e", "1.2.826.0.1.3680043.2.1125." + modification_date + ".1" + modification_time),
            ("0020|0013", str(idx)),
            ("0020|0011", str(series_number))
        ]

        tags_to_copy = [  # "0010|0010",  # Patient Name
            "0010|0020",  # Patient ID
            "0010|0030",  # Patient Birth Date
            "0020|000d",  # Study Instance UID, for machine consumption
            "0020|0010",  # Study ID, for human consumption
            "0008|0020",  # Study Date
            "0008|0030",  # Study Time
            "0008|0050",  # Accession Number
            "0008|0060",  # Modality,
            "0020|0032",
            "0020|1041",
            "0008|1030"  # study description
        ]

        for tag in tags_to_copy:
            if tag in image.GetMetaDataKeys():
                try:
                    final.SetMetaData(tag, image.GetMetaData(tag))
                except:
                    final.SetMetaData(tag, "UNREADABLE")

        list(map(lambda tag_value: final.SetMetaData(tag_value[0], tag_value[1]), series_tag_values))

        if series_number > 1:
            final.SetMetaData("0008|0008", "DERIVED\\SECONDARY")

        if not os.path.exists(folder_name):
            os.makedirs(folder_name)

        writer.SetFileName(folder_name + "/%s_%06d.dcm" % (prefix, idx))
        writer.Execute(final)


def get_mesh_plot(poly):
    plot = k3d.plot(height=400, lighting=1.75)
    plot.menu_visibility = False
    plot.axes_helper = False

    for mesh in poly.values():
        p = k3d.mesh(vertices=mesh.vertices, indices=mesh.indices, color=mesh.color, name=mesh.name,
                     opacity=mesh.opacity, flat_shading=False)
        p.hover_callback = mesh.hover_callback

        p.model_matrix = k3d.transform(translation=np.random.random(3) * 2).model_matrix
        plot += p

    plot.grid_visible = False
    plot.grid_auto_fit = True
    plot.depth_peels = 12
    plot.camera_auto_fit = False
    plot.camera = plot.get_auto_camera(0.8, -35, 70)

    return plot


def get_3d_plot(scan, masks, poly, size_reduce=[1, 1, 1], poly3d=None):
    plot = k3d.plot(height=400, background_color=0, lighting=1.75)
    plot.menu_visibility = False
    plot.axes_helper = False

    data = sitk.GetArrayFromImage(scan).astype(np.float32).copy()
    data = data[::size_reduce[0], ::size_reduce[1], ::size_reduce[2]]

    volume_slice = k3d.volume_slice(
        data,
        color_range=[-1024, np.percentile(data, 98)],
        color_map=k3d.matplotlib_color_maps.Binary_r,
        bounds=get_bounds(scan),
        slice_z=data.shape[0] // 2,
        slice_y=data.shape[1] // 2,
        slice_x=data.shape[2] // 2,
        mask=masks[::size_reduce[0], ::size_reduce[1], ::size_reduce[2]],
        active_masks=[],
        mask_opacity=0.5,
        name='scan',
        color_map_masks=[k3d.nice_colors[i % 19] for i in range(256)]
    )

    for mesh in poly.values():
        p = k3d.mesh(vertices=mesh.vertices, indices=mesh.indices, color=mesh.color, name=mesh.name)
        p.visible = False
        plot += p

    plot += volume_slice

    plot.slice_viewer_mask_object_ids = [o.id for o in plot.objects if o.type == 'Mesh']
    plot.grid_visible = False
    plot.slice_viewer_object_id = volume_slice.id
    plot.grid_auto_fit = True
    plot.camera_auto_fit = False

    # plot.hidden_object_ids = plot.object_ids

    if poly3d is None:
        poly3d = poly

    for mesh in poly3d.values():
        p = k3d.mesh(vertices=mesh.vertices, indices=mesh.indices, color=mesh.color,
                     opacity=mesh.opacity, name='mesh_' + mesh.name, flat_shading=False)
        plot += p

    # v = np.vstack([p.vertices for p in poly3d.values()])
    # center = np.mean(v, axis=0).tolist()
    # size = (np.max(v, axis=0) - np.min(v, axis=0)).tolist()
    # plot.camera = [
    #     center[0] + size[0] * 0.85, center[1] - size[1] * 1.35, center[2] + size[2] * 0.3,
    #     center[0], center[1], center[2],
    #     0, 0, 1
    # ]

    return plot


def get_2d_plot(scan, masks, poly, size_reduce=[1, 1, 1]):
    plot = k3d.plot(height=400, background_color=0, camera_mode='slice_viewer')

    plot.menu_visibility = False
    plot.axes_helper = False

    data = sitk.GetArrayFromImage(scan).astype(np.float32).copy()
    data = data[::size_reduce[0], ::size_reduce[1], ::size_reduce[2]]

    volume_slice = k3d.volume_slice(
        data,
        color_range=[-1024, np.percentile(data, 98)],
        color_map=k3d.matplotlib_color_maps.Binary_r,
        color_map_masks=[k3d.nice_colors[i % 19] for i in range(256)],
        bounds=get_bounds(scan), slice_z=data.shape[0] // 2,
        mask=masks[::size_reduce[0], ::size_reduce[1], ::size_reduce[2]],
        active_masks=np.unique(masks).tolist(),
        mask_opacity=0.5,
        name='scan'
    )

    for p in poly.values():
        plot += p

    plot += volume_slice

    plot.slice_viewer_mask_object_ids = [o.id for o in plot.objects if o.type == 'Mesh']
    plot.grid_visible = False
    plot.slice_viewer_object_id = volume_slice.id
    plot.grid_auto_fit = True
    plot.camera_auto_fit = False

    return plot


def rearange_masks(masks, scan):
    masks = sitk.GetArrayFromImage(masks)
    new_masks = np.zeros_like(masks)

    for i, v in enumerate(np.unique(masks)):
        new_masks[masks == v] = i

    masks = sitk.GetImageFromArray(new_masks)
    masks.CopyInformation(scan)

    return masks


def load_unify_image(path=None, image=None):
    if path is not None:
        if type(path) is not list:
            path = [path]
        print('paths', path)
        for p in path:
            p = glob2.glob(p)

            if len(p) == 1:
                image = sitk.ReadImage(p[0])
                break

    image.SetDirection(np.round(np.array(image.GetDirection())))

    direction = np.array(image.GetDirection()).reshape(len(image.GetSize()), -1)
    ind = np.argmax(np.abs(direction), axis=1)

    new_size = np.array(image.GetSize())[ind]# // 2
    new_spacing = np.array(image.GetSpacing())[ind]# * 2
    new_extent = (new_size - np.ones(3)) * new_spacing
    new_origin = np.array(image.GetOrigin()) - new_extent * (np.diag(direction[:, ind]) < 0)

    resample = sitk.ResampleImageFilter()

    resample.SetOutputSpacing(new_spacing.tolist())
    resample.SetSize(new_size.tolist())
    resample.SetOutputDirection(np.eye(3).flatten())
    resample.SetOutputOrigin(new_origin.tolist())

    resample.SetTransform(sitk.Transform())
    resample.SetDefaultPixelValue(image.GetPixelIDValue())
    resample.SetInterpolator(sitk.sitkNearestNeighbor)

    # pad to divide by 4
    im = resample.Execute(image)
    pad = [x % 4 for x in im.GetSize()]
    im = sitk.ConstantPad(im, [0, 0, 0], pad)
    im.SetOrigin((0, 0, 0))

    return im


def scale_lightness(rgb, scale_l):
    r, g, b = (rgb // 256 // 256 % 256 / 255.0, rgb // 256 % 256 / 255.0, rgb % 256 / 255.0)
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    r, g, b = colorsys.hls_to_rgb(h, min(1, l * scale_l), s=s)

    return (int(r * 255.0) << 16) + (int(g * 255.0) << 8) + int(b * 255.0)


def get_id_by_name(name, d, key='Structure'):
    for k, i in d.items():
        if i[key] == name:
            return int(k)


def get_bounds(img):
    origin = img.GetOrigin()
    size = np.array(img.GetSpacing()) * np.array(img.GetSize())

    return np.array([origin[0], origin[0] + size[0],
                     origin[1], origin[1] + size[1],
                     origin[2], origin[2] + size[2]])


def get_volume(img):
    data = sitk.GetArrayFromImage(img).astype(np.float32)

    return k3d.volume(data, bounds=get_bounds(img),
                      samples=512, color_map=k3d.paraview_color_maps.Jet, shadow='dynamic',
                      shadow_delay=250)


def resample(image, size):
    new_spacing = np.array(image.GetSpacing()) * (np.array(image.GetSize()) / np.array(size))

    return sitk.Resample(image, size, sitk.Transform(), sitk.sitkLinear, image.GetOrigin(),
                         new_spacing, image.GetDirection(), 0.0, image.GetPixelIDValue())


def contour(data, bounds, values, quantize_points_factor=0.0, clustering_factor=6):
    vti = vtk.vtkImageData()
    vti.SetOrigin(bounds[::2])
    vti.SetDimensions(np.array(data.shape[::-1]))
    vti.SetSpacing((bounds[1::2] - bounds[::2]) / np.array(data.shape[::-1]))

    arr = numpy_support.numpy_to_vtk(num_array=data.ravel(), deep=True, array_type=vtk.VTK_FLOAT)
    arr.SetName('d')
    vti.GetPointData().AddArray(arr)
    vti.GetPointData().SetActiveScalars('d')

    contour = vtk.vtkImageMarchingCubes()
    contour.SetInputData(vti)
    contour.SetNumberOfContours(values.shape[0])

    for i, v in enumerate(values):
        contour.SetValue(i, v)
    contour.ComputeScalarsOn()
    contour.Update()

    quantize = vtk.vtkQuantizePolyDataPoints()
    quantize.SetQFactor(quantize_points_factor)

    quantize.SetInputConnection(contour.GetOutputPort())
    quantize.Update()

    if clustering_factor > 0:
        mesh = pyvista.PolyData(quantize.GetOutput())
        mesh.compute_normals(inplace=True)
        clus = pyacvd.Clustering(mesh)
        clus.cluster(mesh.n_points // clustering_factor)

        try:
            poly = clus.create_mesh()
        except:
            poly = quantize.GetOutput()
    else:
        poly = quantize.GetOutput()

    return poly


# add by mulder
def flipMsk12(img):
    return sitk.GetImageFromArray(np.flip(np.flip(sitk.GetArrayFromImage(img), 1), 2))
