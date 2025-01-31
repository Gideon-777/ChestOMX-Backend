import argparse
import base64
import json
import k3d_pro as k3d
import os
import re
import webpage2html
from docxtpl import DocxTemplate
from mako.template import Template

from plots import *


def versiontuple(v):
    return tuple(map(int, (v.split("."))))


assert versiontuple(k3d.__version__) >= versiontuple("2.12.0")

parser = argparse.ArgumentParser(description='Produce PhenoMx IPF report')
parser.add_argument('--header_name', help='case name', default='')
parser.add_argument('--dicom', help='whether a dicom should be added', action='store_true')
parser.add_argument('--pdf', help='whether a pdf should be added', action='store_true')
parser.add_argument('--mask_flip', help='whether a mask should be flipped', action='store_true')
parser.add_argument('--lobes3d', help='whether a lobes 3d section should be added',
                    action='store_true')
# parser.add_argument('--lobes_ipf_3d', help='whether a lobes+IPF 3d section should be added',
#                     action='store_true')
parser.add_argument('--lobes', help='whether a lobes section should be added', action='store_true')
parser.add_argument('--lungs', help='whether a lungs section should be added', action='store_true')
parser.add_argument('--ipf',
                    help='whether a ipf section should be added',
                    action='store_true')
parser.add_argument('--summary', help='whether a summary section should be added',
                    action='store_true')
parser.add_argument('--reduce_x', help='Reduce resolution x', type=int, default=1)
parser.add_argument('--reduce_y', help='Reduce resolution y', type=int, default=1)
parser.add_argument('--reduce_z', help='Reduce resolution z', type=int, default=1)
parser.add_argument('--port', help='Port for headless', type=int, default=8080)
parser.add_argument('--exam', help='name of case', required=True)
parser.add_argument('--out', help='filename of output file', default='')

args = parser.parse_args()

IPF_map = {
    "1": "Normal",
    "2": "GroundGlass",
    "3": "Honeycomb",
    "4": "Reticular",
    "5": "ModerateLAA",
    "6": "MildLAA",
    "7": "SevereLAA"
}

variables = {}
series_number = 2
plots = []
filename = args.exam if args.exam != '' else args.exam
variables['case'] = args.header_name if args.header_name != '' else args.exam

variables['dicom'] = args.dicom
variables['lobes3d'] = args.lobes3d
variables['lobes'] = args.lobes
variables['lungs'] = args.lungs
# variables['lobes_ipf_3d'] = args.lobes_ipf_3d
variables['ipf'] = args.ipf
variables['pdf'] = args.pdf
# variables['ggo_consolidation'] = args.ggo_consolidation

# image loading
scan = load_unify_image(['./input/' + args.exam + '/*_0000.nii.gz', './input/' + args.exam + '/*_scan.nii.gz'])

dicoms = None
size_reduce = [args.reduce_z, args.reduce_y, args.reduce_x]
print(scan.GetSize())
scan.SetOrigin((0, 0, 0))
voxel_volume = np.prod(np.array(scan.GetSpacing())) / (10 ** 3)

if os.path.isdir('./input/' + args.exam + '/dicom'):
    reader = sitk.ImageSeriesReader()
    reader.MetaDataDictionaryArrayUpdateOn()
    reader.LoadPrivateTagsOn()

    dicom_names = reader.GetGDCMSeriesFileNames('./input/' + args.exam + '/dicom')
    dicoms = [sitk.ReadImage(f) for f in dicom_names]

if args.dicom and dicoms is not None:
    print('dicom')
    reader = dicoms[0]

    # many magic strings - it's a dicom standard - https://dicom.innolitics.com/ciods/rt-plan/patient/00100010
    meta = {
        "patientName": reader.GetMetaData('0010|0010') \
            if '0010|0010' in reader.GetMetaDataKeys() else '',
        "patientId": reader.GetMetaData('0010|0020') \
            if '0010|0020' in reader.GetMetaDataKeys() else '',
        "patientBirthdate": reader.GetMetaData('0010|0030') \
            if '0010|0030' in reader.GetMetaDataKeys() else '',
        "patientSex": reader.GetMetaData('0010|0040') \
            if '0010|0040' in reader.GetMetaDataKeys() else '',
        "patientAge": reader.GetMetaData('0010|1010') \
            if '0010|1010' in reader.GetMetaDataKeys() else '',
        "studyDescription": reader.GetMetaData('0008|1030') \
            if '0008|0030' in reader.GetMetaDataKeys() else '',
        "institutionName": reader.GetMetaData('0008|0080') \
            if '0008|0080' in reader.GetMetaDataKeys() else '',
        "institutionAddress": reader.GetMetaData('0008|0081') \
            if '0008|0081' in reader.GetMetaDataKeys() else '',
        "stationName": reader.GetMetaData('0008|1010') \
            if '0008|1010' in reader.GetMetaDataKeys() else '',
        "manufacturer": reader.GetMetaData('0008|1090') \
            if '0008|1090' in reader.GetMetaDataKeys() else '',
        "sliceThickness": reader.GetMetaData('0018|0050') \
            if '0018|0050' in reader.GetMetaDataKeys() else ''
    }

    variables = {**variables, **meta}

if args.lungs:
    print('lungs')
    lungs = load_unify_image(['./input/' + args.exam + '/*_lungs.nii.gz', './input/' + args.exam + '/*_lung.nii.gz'])
    if args.mask_flip:
        lungs = flipMsk12(lungs)
    lungs.SetOrigin((0, 0, 0))
    lungs.SetSpacing(scan.GetSpacing())

    poly_lungs_smoothed, masks_lungs, lungs_volumes = get_poly(lungs, 'Lungs', size_reduce, 18)
    poly_lungs, masks_lungs, lungs_volumes = get_poly(lungs, 'Lungs', size_reduce, 2)

    lungs_volumes['1.0']['Label'] = "Left lung"
    lungs_volumes['2.0']['Label'] = "Right lung"

    lungs_plot = get_3d_plot(scan, masks_lungs, poly_lungs, size_reduce, poly_lungs_smoothed)

    plots.append((lungs_plot, 'lungs', 'volumeViewer'))

    variables['lungs_volumes'] = [lungs_volumes[i] for i in sorted(lungs_volumes.keys())]

    if dicoms is not None and variables['dicom']:
        save_dicom('lungs', dicoms, filename, scan, masks_lungs, series_number)
        series_number += 1

if args.lobes or args.lobes3d:
    print('lobes common')
    lobes = load_unify_image(['./input/' + args.exam + '/*_lobe.nii.gz', './input/' + args.exam + '/*_lobes.nii.gz'])

    if args.mask_flip:
        lobes = flipMsk12(lobes)
    lobes.SetOrigin((0, 0, 0))
    lobes.SetSpacing(scan.GetSpacing())
    poly_lobes_smoothed, masks_lobe, lobes_volumes = get_poly(lobes, 'Lobes', size_reduce, 18)

    lobes_volumes['1.0']['Label'] = "Left upper lobe"
    lobes_volumes['2.0']['Label'] = "Left lower lobe"
    lobes_volumes['3.0']['Label'] = "Right upper lobe"
    lobes_volumes['4.0']['Label'] = "Right middle lobe"
    lobes_volumes['5.0']['Label'] = "Right lower lobe"

    variables['lobes_volumes'] = [lobes_volumes[i] for i in sorted(lobes_volumes.keys())]

    if dicoms is not None and variables['dicom']:
        save_dicom('lobes', dicoms, filename, scan, masks_lobe, series_number)
        series_number += 1

if args.lobes3d:
    print('lobes3d')
    lobes3d_plot = get_mesh_plot(poly_lobes_smoothed)

    plots.append((lobes3d_plot, 'lobes3d', 'meshViewer'))

if args.lobes:
    print('lobes')
    poly_lobes, masks_lobe, lobes_volumes = get_poly(lobes, 'Lobes', size_reduce, 8)
    lobes_plot = get_3d_plot(scan, masks_lobe, poly_lobes, size_reduce, poly_lobes_smoothed)

    plots.append((lobes_plot, 'lobes', 'volumeViewer'))

if args.ipf and (args.lobes or args.lobes3d):
    print('ipf')
    IPF = load_unify_image(
        ['./input/' + args.exam + '/*_7label.nii.gz', './input/' + args.exam + '/*_infiltration.nii.gz']
    )
    if args.mask_flip:
        IPF = flipMsk12(IPF)
    IPF.SetOrigin((0, 0, 0))
    IPF.SetSpacing(scan.GetSpacing())

    poly_IPF, masks_IPF, IPF_volumes = get_poly(IPF, 'IPF', size_reduce, 8, names=IPF_map)
    IPF_plot = get_3d_plot(scan, masks_IPF, poly_IPF, size_reduce)

    plots.append((IPF_plot, 'IPF', 'volumeViewer'))

    variables['IPF_volumes'] = [IPF_volumes[i] for i in sorted(IPF_volumes.keys())]
    variables['IPF'] = variables['IPF_volumes']

    if dicoms is not None and variables['dicom']:
        save_dicom('IPF', dicoms, filename, scan, masks_IPF, series_number)
        series_number += 1

if args.ipf and (args.lobes or args.lobes3d) and args.summary:
    print('summary')
    variables['summary'] = []

    for l in range(1, 6):
        row = {}
        lobe_volume = np.sum(masks_lobe == l) * voxel_volume

        for ipf_id, ipf_name in IPF_map.items():
            row[IPF_map[ipf_id]] = np.sum((masks_lobe == l) * (masks_IPF == int(ipf_id))) * voxel_volume

        lobe = {
            'Label': lobes_volumes[str(l) + '.0']['Label'],
            'Volume': lobe_volume,
            'volumes': row
        }

        variables['summary'].append(lobe)

# if args.lobes_ipf_3d and (args.lobes or args.lobes3d) and args.ipf:
#     print('lobes_ipf_3d')
#
#     poly_IPF, _, _ = get_poly(IPF, 'Infiltration', 2)
#
#     for p in poly_IPF.values():
#         p.hover_callback = True
#         p.color = 0xff0000
#
#     for p in poly_lobes_smoothed.values():
#         p.opacity = 0.3
#
# label = k3d.label("test", name="label", is_html=True, color=0, size=2.0, mode='local')
# label.visible = False
#
#     lobes_infiltration_3d_plot = get_mesh_plot({**poly_lobes_smoothed, **poly_IPF})
#     lobes_infiltration_3d_plot += label
#     lobes_infiltration_3d_plot.mode = 'callback'
#
#     plots.append((lobes_infiltration_3d_plot, 'lobes_infiltration_3d', 'meshViewer'))

# if args.ggo_consolidation and (args.lobes or args.lobes3d):
#     print('ggo_consolidation')
#     ggo_consolidation, _ = load_unify_image(
#         './input/ggo_consolidation/' + args.exam + '*.nii.gz')
#     ggo_consolidation.SetOrigin((0, 0, 0))
#     ggo_consolidation.SetSpacing(scan.GetSpacing())
#
#     poly_ggo_consolidation, masks_ggo_consolidation, ggo_consolidation_volumes = \
#         get_poly(ggo_consolidation, 'GGO_Consolidation', 2)
#
#     ggo_consolidation_plot = get_3d_plot(scan, masks_ggo_consolidation, poly_ggo_consolidation,
#                                          size_reduce)
#
#     plots.append((ggo_consolidation_plot, 'ggo_consolidation', 'volumeViewer'))
#
#     variables['ggo'] = {
#         'LeftGGO': {
#             'Label': 'Left ggo',
#             'Volume': np.sum((masks_lobe < 3) * (masks_ggo_consolidation == 2)) * voxel_volume
#         },
#         'RightGGO': {
#             'Label': 'Right ggo',
#             'Volume': np.sum((masks_lobe >= 3) * (masks_ggo_consolidation == 2)) * voxel_volume
#         },
#         'LeftConsolidation': {
#             'Label': 'Left consolidation',
#             'Volume': np.sum((masks_lobe < 3) * (masks_ggo_consolidation == 1)) * voxel_volume
#         },
#         'RightConsolidation': {
#             'Label': 'Right consolidation',
#             'Volume': np.sum((masks_lobe >= 3) * (masks_ggo_consolidation == 1)) * voxel_volume
#         }
#     }

# Template
filename = args.out if args.out != '' else args.exam

template = Template(filename='./template/index.html')
empty_snapshot = k3d.plot().get_snapshot(9)

with open(filename + '.json', 'w') as f:
    f.write(json.dumps(variables, indent=4))

    # PDF
    if variables['pdf']:
        screenshots = {}

        for plot, name, _ in plots:
            screenshots = {**screenshots, **get_screenshots(plot, name.capitalize(), args.port)}

        # dump
        for k, v in screenshots.items():
            with open(k + ".png", "wb") as f:
                f.write(v)

        doc = DocxTemplate("./template/template.docx")

        for plot, name, _ in plots:
            key = name.capitalize()
            doc.pics_to_replace[key + '1'] = screenshots[key + '_z']
            doc.pics_to_replace[key + '2'] = screenshots[key + '_x']
            doc.pics_to_replace[key + '3'] = screenshots[key + '_y']

            if key + '_3d' in screenshots:
                doc.pics_to_replace[key + '4'] = screenshots[key + '_3d']

        doc.render(variables)

        try:
            doc.pre_processing()
        except Exception:
            # accept missing images (due to conditional sections)
            pass

        doc.docx.save(filename + ".docx")
        doc.post_processing(filename + ".docx")

        try:
            from docx2pdf import convert

            convert(filename + ".docx")
        except Exception:
            try:
                subprocess.call("libreoffice", "--headless --convert-to pdf " + filename + ".docx")
            except Exception:
                pass

# HTML
first = 1
for i, _, _ in plots:
    for obj in i.objects:
        if obj.name == 'scan':
            if first == 1:
                first = 0
                obj.name = 'scan_data'
            else:
                obj.volume[:, :, :] = 0

snapshot = empty_snapshot
FFLATE_JS = re.findall(r'id=\'fflatejs\'>(.+?)</script>', snapshot, re.MULTILINE | re.DOTALL)[0]
REQUIRE_JS = re.findall(r'id=\'requirejs\'>(.+?)</script>', snapshot, re.MULTILINE | re.DOTALL)[0]
K3D_SOURCE = re.findall(r'window.k3dCompressed = \'(.+?)\'', snapshot, re.MULTILINE | re.DOTALL)[0]

variables['FFLATE_JS'] = FFLATE_JS
variables['REQUIRE_JS'] = REQUIRE_JS
variables['K3D_SOURCE'] = K3D_SOURCE

widgets_data = {
    'plots': {}
}

for plot, name, type in plots:
    plot.minimum_fps = -1
    snapshot = plot.get_binary_snapshot(1)
    print(name, len(snapshot) / (1024 ** 2))
    k3d_data = base64.b64encode(snapshot).decode("utf-8")

    widgets_data['plots'][name] = {
        'data': k3d_data,
        'type': type
    }

variables['json_widgets_data'] = json.dumps(widgets_data)
variables['json_widgets_data'] = json.dumps(widgets_data)

with open('./template/temp_' + filename + '.html', 'wb') as f:
    f.write(template.render_unicode(**variables).encode())

baked_template = webpage2html.generate('./template/temp_' + filename + '.html', comment=False,
                                       keep_script=True)

# cdata fix after webpage2html
baked_template = baked_template.replace('&gt;', '>').replace('&lt;', '<')

with open(filename + '.html', 'wb') as f:
    f.write(baked_template.encode())

os.remove('./template/temp_' + filename + '.html')
#
# pdfkit.from_url(filename + '.html', filename + '.pdf', options={
#     "window-status": "ready_to_print",
#     "print-media-type": None
# })