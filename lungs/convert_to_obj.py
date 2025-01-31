import nibabel as nib
import numpy as np
from skimage import measure
import skimage.transform as transform
import skimage.filters as filters
import skimage.morphology as morphology
from scipy import ndimage
import sys

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

    overall_map = { label: i+1 for i, label in enumerate(labels) }

    img = nib.load(nii_image_path)

    arr = img.get_fdata()

    affine = np.abs(np.diag(img.affine)[:3])
    sizes = img.shape

    print(sizes)

    output_shape = (sizes * affine).astype('uint16')

    print(output_shape)

    arr = transform.resize(arr, output_shape=output_shape, order=0, mode='edge', anti_aliasing=False)
    print(np.unique(arr))

    print(np.unique(arr))

    x, y, z = np.nonzero(arr)

    verts, faces, normals, values = measure.marching_cubes(arr, 0)


    faces = faces+1

    print(len(verts), len(faces), len(normals), len(values))
    mtl_file_path = obj_file_path.replace('.obj', '.mtl')
    obj_file = open(obj_file_path, 'w')
    obj_file.write('# OBJ file\n')
    obj_file.write(f'mtllib {mtl_file_path}\n')

    for item in verts:
        obj_file.write("v {0} {1} {2}\n".format(item[0], item[1], item[2]))

    for item in normals:
        obj_file.write("vn {0} {1} {2}\n".format(item[0],item[1],item[2]))

    print(np.unique(values))

    for key, val in overall_map.items():
        if val == 0: continue
        faces1 = [ face for face in faces if values[face[0]-1] == val ]
        obj_file.write(f'o {key}\n')
        obj_file.write(f'usemtl {key}\n')

        for item in faces1:
            obj_file.write("f {0} {1} {2}\n".format(item[0],item[1],item[2]))  

    obj_file.close()

    mtl_file = open(mtl_file_path, 'w')

    for key, val in overall_map.items():
        if val == 0: continue
        mtl_file.write(f"newmtl {key}\n")
        color = all_colors[val-1]
        color = ' '.join(list(map(lambda x: str(x/255).ljust(8, '0')[:8], color)))
        mtl_file.write(f"Kd {color}\n")
        mtl_file.write(f"Ka {color}\n")
        mtl_file.write(f"Ks {color}\n")
        mtl_file.write("\n")

    mtl_file.close()

if __name__ == '__main__':

    convert('../outputs/lungs/01062933_fa270fb13e624fc7b51e3a3bd6fc7c80.nii.gz', '../objs/lungs/01062933_fa270fb13e624fc7b51e3a3bd6fc7c80.obj', ['LL', 'RL'])