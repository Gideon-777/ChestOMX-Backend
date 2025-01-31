import nibabel as nib
import numpy as np
import sys
import vtk
import vtk.util.numpy_support
import math


all_colors = np.array([
    [128,174,128],
    [241,214,145],
    [177,122,101],
    [111,184,210],
    [216,101,79],
    [221,130,101],
    [144,238,144],
    [192,104,88],
    [220,245,20],
    [78,63,0],
    [255,250,220],
    [230,220,70],
    [200,200,235],
    [250,250,210],
    [244,214,49],
    [0,151,206],
    [216,101,79],
    [183,156,220],
    [221,130,101],
    [152,189,207],
    [111,184,210],
    [178,212,242],
    [68,172,100],
    [111,197,131],
    [85,188,255],
    [0,145,30],
    [214,230,130],
    [78,63,0],
    [218,255,255],
    [170,250,250],
    [140,224,228],
    [188,65,28],
    [216,191,216],
    [145,60,66],
    [150,98,83],

])
def convert(nii_image_path, obj_file_path, labels):


    img = nib.load(nii_image_path)

    arr = img.get_fdata()

    affine = img.header.get_zooms()
    sizes = img.shape[::-1]
    affine = affine[::-1]

    # output_shape = (sizes * affine).astype('uint16')

    overall_map = {
        label: i for i, label in enumerate(labels)
    }

    all_verts = []
    all_faces = []
    num_verts = 0


    for i in range(1, len(labels)):

        arr_i = (arr == i).astype('uint8')

        if arr_i.sum() < 1: 
            all_faces.append(0)
            all_verts.append(0)
            continue # if no pixel of this label, don't make 3D

        vtk_image = vtk.vtkImageData()
        vtk_image.SetDimensions(*sizes)
        vtk_image.SetSpacing(*affine)
        vtk_image.SetOrigin(0, 0, 0)
        vtk_image.AllocateScalars(vtk.VTK_UNSIGNED_CHAR, 1)

        vtk_arr = vtk.util.numpy_support.numpy_to_vtk(arr_i.ravel(), deep=True)
        vtk_arr.SetName(labels[i])

        vtk_image.GetPointData().AddArray(vtk_arr)
        vtk_image.GetPointData().SetActiveScalars(labels[i])

        contour = vtk.vtkMarchingCubes() # convert pixel to 3D pionts (triangles)
        contour.SetInputData(vtk_image)
        contour.SetNumberOfContours(1)
        contour.SetValue(1, 1)
        contour.Update()

        smoothing_filter = vtk.vtkWindowedSincPolyDataFilter() # smoothen the points
        smoothing_filter.SetInputConnection(contour.GetOutputPort())
        smoothing_filter.SetNumberOfIterations(20)
        smoothing_factor = 0.2
        pass_band = math.pow(10.0, -4.0 * smoothing_factor)
        smoothing_filter.SetPassBand(pass_band)
        smoothing_filter.SetBoundarySmoothing(False)
        smoothing_filter.Update()

        output = smoothing_filter.GetOutput()

        verts = vtk.util.numpy_support.vtk_to_numpy(output.GetPoints().GetData())
        # norms = vtk.util.numpy_support.vtk_to_numpy(output.GetPointData().GetNormals())
        faces = vtk.util.numpy_support.vtk_to_numpy(output.GetPolys().GetData())

        faces = faces.reshape((-1, 4))[:, 1:]
        faces = faces+1

        all_faces.append(faces + num_verts)
        all_verts.append(verts)
        num_verts += len(verts)

    mtl_file_path = obj_file_path.replace('.obj', '.mtl')
    obj_file = open(obj_file_path, 'w')
    obj_file.write('# OBJ file\n')
    obj_file.write(f'mtllib {mtl_file_path}\n')

    for verts in all_verts:
        if isinstance(verts, int) and verts == 0: continue
        for item in verts:
            obj_file.write("v {0} {1} {2}\n".format(item[0], item[1], item[2]))

    for key, val in overall_map.items():
        if val == 0: continue
        faces = all_faces[val - 1]
        obj_file.write(f'o {key}\n')
        obj_file.write(f'usemtl {key}\n')
        if isinstance(faces, int) and faces == 0: continue
        for item in faces:
            obj_file.write("f {0} {1} {2}\n".format(item[0], item[1], item[2]))  

    obj_file.close()

    mtl_file = open(mtl_file_path, 'w')

    for key, val in overall_map.items():
        if val == 0: continue
        mtl_file.write(f"newmtl {key}\n")
        color = all_colors[val-1]
        color = ' '.join(list(map(lambda x: str(x/255).ljust(8, '0')[:8], color)))
        color_0 = ' '.join(list(map(lambda x: str(x/255).ljust(8, '0')[:8], [0, 0, 0])))

        mtl_file.write(f"Kd {color}\n")
        mtl_file.write(f"Ka {color_0}\n")
        mtl_file.write(f"Ks {color_0}\n")
        mtl_file.write("\n")

    mtl_file.close()


if __name__ == '__main__':
    print('hello')
