import argparse
import os
import dicom2nifti
from skimage.measure import label, regionprops
import multiprocessing
import pandas as pd
import nibabel
from pydicom.uid import ExplicitVRLittleEndian
import numpy as np
import sys
import dicom2nifti.settings as settings
import SimpleITK as sitk
import pydicom
import time
import math
import tempfile
import uuid
import redis
import pickle

import firebase_admin
from firebase_admin import firestore
import pyrebase
import json
import datetime
import yaml
import skimage.transform as transform
import shutil

from nnunet_utils import resize_segmentation
import generic_UNet
import torch

import utils
import convert_to_obj
import convert_to_obj2

import make_reports


def get_metrics(label, pred):
    tp = (label * pred).sum()
    fp = ((label == 0) * (pred == 1)).sum()
    fn = ((label == 1) * (pred == 0)).sum()
    recall = tp / (tp + fn + 1e-12)
    precision = tp / (tp + fp + 1e-12)
    dice = 2 * recall * precision / (recall + precision + 1e-12)
    jaccard = tp / (tp + fp + fn + 1e-12)
    return recall, precision, dice, jaccard

def ctss_results_map(involvmenets, ctss):
    ctss_results = {
            'involvement_rul': involvmenets[0],
            'involvement_rml': involvmenets[1],
            'involvement_rll': involvmenets[2],
            'involvement_lul': involvmenets[3],
            'involvement_lll': involvmenets[4],
            'ctss_rul': ctss[0],
            'ctss_rml': ctss[1],
            'ctss_rll': ctss[2],
            'ctss_lul': ctss[3],
            'ctss_lll': ctss[4],
    }
    return ctss_results

# involvement to ctss score
#def get_ctss(inv):
#    if inv < 0.1:
#        return 0
#    elif inv < 5:
#        return 1
#    elif inv < 25:
#        return 2
#    elif inv < 50:
#        return 3
#    elif inv < 75:
#        return 4
#    return 5


def get_ctss(inv, ctss_map):
    for key, value in ctss_map.items():
        if inv >= value[0] and inv <= value[1]:
            return key



# calculate involvement and ctss
def calc_ctss(lobes_path, infiltration_path, ctss_map):
    lobes_arr = nibabel.load(lobes_path).get_fdata()
    infiltration_arr = nibabel.load(infiltration_path).get_fdata()
    involvements = []
    ctss = []
    for i in range(1, 6): # go through each lobe and get the involvement % in that lobe and corresponding ctss score
        involvement = ((lobes_arr == i) * infiltration_arr).sum() / (lobes_arr == i).sum() * 100.0
        involvements.append(involvement)
        ctss.append(get_ctss(involvement, ctss_map))
    return involvements, ctss

def maybe_mkdir(path):
    if not os.path.exists(path):
        os.mkdir(path)


def extract_labels(arr, bbox):
    ids, cnts = np.unique(
        arr[bbox[0] : bbox[3], bbox[1] : bbox[4], bbox[2] : bbox[5]], return_counts=True
    )
    ids = list(ids)
    cnts = list(cnts)
    if 0 in ids:
        idx = ids.index(0)
        del cnts[idx]
        del ids[idx]
    return ids[cnts.index(max(cnts))]


# Lungs Postprocessing
def convert_lungs(lung_folder, output_folder, image):

    # lung_folder = args.lung_output
    # output_folder = args.lung_output
    img_lung = nibabel.load(f"{lung_folder}/{image}")
    arr_lung1 = img_lung.get_fdata()
    arr_lung = arr_lung1.copy()
    arr_lung[arr_lung > 0] = 1  # take the overall mask of the lungs as a whole
    L = label(arr_lung)

    print("Number of lungs: {}".format(np.max(L)))

    props = regionprops(L)
    areas = [
        (i, prop.area) for i, prop in enumerate(props) if prop.area > 30000
    ]  # Extract all the individual,set a threshold to 500000 so as to remove small portions
    if len(areas) == 0:
        areas = [(i, prop.area) for i, prop in enumerate(props)]
    areas = sorted(
        areas, key=lambda x: x[1], reverse=True
    )  # arrange the individual segments and sort them so as to remove highest two segments
    indices = list(zip(*areas))[0]
    correct_centroids = np.array(
        [props[i].centroid for i in indices[:2]]
    )  # extract the centroid of two individual highest area segments which is right and left lung individually

    modified_lung = np.zeros_like(arr_lung)  # create an empty array
    for i in range(len(correct_centroids)):  # original 2 segments properly segmented
        modified_lung[L == props[indices[i]].label] = arr_lung[
            L == props[indices[i]].label
        ]  # store the two largest  areas in the empty array

    modified_lung = (
        arr_lung1 * modified_lung
    )  # remove unneccasry regions outside the lung region
    arr = modified_lung.copy()

    L = label(arr)
    props = regionprops(L)

    areas = [
        (i, extract_labels(arr.copy(), prop.bbox), prop.area, prop.centroid)
        for i, prop in enumerate(props)
    ]  # store the areas of individual segments within the lungregion,(modified the function to look for max pixel counts within the segmented region instead of centorid)

    areas_df = pd.DataFrame(areas, columns=["id", "lung_id", "area", "centroid"])
    areas_df["max_area"] = areas_df.groupby("lung_id")["area"].transform(max)
    area_df = areas_df[areas_df["area"] == areas_df["max_area"]]
    area_df = area_df[area_df["lung_id"] != 0]
    correct_centroids = np.array([props[i].centroid for i in list(area_df.index)])
    areas_df = areas_df.drop(area_df.index).reset_index(drop=True)
    area_df = area_df.reset_index(drop=True)

    print("Number of incorrect segments: {}".format(len(areas_df)))
    modified_lungs = np.zeros_like(arr)
    for i in range(len(area_df)):  # this for loop to store the largest two lung regions
        # print(i)
        modified_lungs[L == props[area_df.iloc[i].id].label] = arr[
            L == props[area_df.iloc[i].id].label
        ]
    for i in range(len(areas_df)):  # rest of segments attached to closest lung
        centroid = props[areas_df.iloc[i].id].centroid
        if areas_df.iloc[i].area < 100: continue
        closest_index = np.argmin(
            ((correct_centroids - centroid) ** 2).sum(1)
        )  # looks into the rest closest and NOT the 5 main closest...
        # print(closest_index)
        modified_lungs[L == props[areas_df.iloc[i].id].label] = arr[
            L == props[area_df.iloc[closest_index].id].label
        ].flatten()[0]

    nibabel.save(
        nibabel.Nifti1Image(modified_lungs, img_lung.affine), f"{output_folder}/{image}"
    )


def lungs_postprocessing():
    try:
        lung_folder = args.lung_output
        images = [
            image for image in os.listdir(lung_folder) if image.endswith(".nii.gz")
        ]

        with multiprocessing.Pool(int(args.num_workers)) as p:
            p.map(convert_lungs, images)
    except Exception as e:
        print("Error in lungs postprocessing {}".format(e))
        exit()


# Lobes postprocessing
def convert_lobes(lung_folder, lobes_folder, image):

    # lobes_folder = args.lobe_output
    # lung_folder = args.lung_output
    # output_folder = args.lobe_output

    output_folder = lobes_folder

    img_lobe = nibabel.load(f"{lobes_folder}/{image}")
    arr_lobe = img_lobe.get_fdata()

    img_lung = nibabel.load(f"{lung_folder}/{image}")
    arr_lung = img_lung.get_fdata()
    arr_lung[arr_lung > 0] = 1  # make left and right lung as one label
    arr = arr_lobe * (
        arr_lung > 0
    )  # multiply lobe and lung results to avoid the unnecessary regions outside lung that come from nnunet predictions
    arr_diff = arr_lung - (
        arr > 0
    )  # extract those regions that are part of lung but not part of lobes results
    L = label(arr)
    props = regionprops(L)

    areas = [
        (i, arr[tuple(map(int, prop.centroid))], prop.area, prop.centroid)
        for i, prop in enumerate(props)
    ]  # store the areas of individual segments within the lobe region along with their centroids
    areas_df = pd.DataFrame(
        areas, columns=["id", "lobe_id", "area", "centroid"]
    )  # Make a dataframe of all individual segments
    areas_df["max_area"] = areas_df.groupby("lobe_id")["area"].transform(max)
    area_df = areas_df[
        areas_df["area"] == areas_df["max_area"]
    ]  # Make a dataframe of correct 5 lobe segments
    area_df = area_df[area_df["lobe_id"] != 0]
    correct_centroids = np.array(
        [props[i].centroid for i in list(area_df.index)]
    )  # Extract centroids of correct 5 segments
    areas_df = areas_df.drop(area_df.index).reset_index(drop=True)
    area_df = area_df.reset_index(drop=True)

    modified_lobes = np.zeros_like(arr_lobe)
    # print(len(area_df), len(areas_df))
    for i in range(len(area_df)):  # original 5 segments properly segmented
        # print(i)
        modified_lobes[L == props[area_df.iloc[i].id].label] = arr_lobe[
            L == props[area_df.iloc[i].id].label
        ]
    for i in range(len(areas_df)):  # rest of segments attached to closest lobe
        centroid = props[areas_df.iloc[i].id].centroid
        closest_index = np.argmin(
            ((correct_centroids - centroid) ** 2).sum(1)
        )  # looks into the rest closest and NOT the 5 main closest...
        # print(closest_index)
        modified_lobes[L == props[areas_df.iloc[i].id].label] = arr_lobe[
            L == props[area_df.iloc[closest_index].id].label
        ].flatten()[0]

    L = label(modified_lobes)
    props = regionprops(L)
    areas = [
        (i, arr[tuple(map(int, prop.centroid))], prop.area, prop.centroid)
        for i, prop in enumerate(props)
    ]  ##Make a dataframe of all individual segments after correcting all lobe segments
    areas_df = pd.DataFrame(areas, columns=["id", "lobe_id", "area", "centroid"])
    areas_df["max_area"] = areas_df.groupby("lobe_id")["area"].transform(max)
    area_df = areas_df[
        areas_df["area"] == areas_df["max_area"]
    ]  # Make a dataframe of correct 5 lobe segments
    area_df = area_df[area_df["lobe_id"] != 0]
    correct_centroids = np.array(
        [props[i].centroid for i in list(area_df.index)]
    )  # Extract centroids of correct 5 segments
    areas_df = areas_df.drop(area_df.index).reset_index(drop=True)
    area_df = area_df.reset_index(drop=True)

    # assign lobe labels to those regions which are not part of nnunet lobe predictions but part of lung predictions
    if np.sum(arr_diff) > 0:
        L = label(arr_diff)
        props = regionprops(L)
        areas = [
            (i, arr_diff[tuple(map(int, prop.centroid))], prop.area, prop.centroid)
            for i, prop in enumerate(props)
            if prop.area > 500
        ]
        if len(areas) > 0:
            areas_df = pd.DataFrame(
                areas, columns=["id", "lobe_id", "area", "centroid"]
            )
            for i in range(len(areas_df)):
                # print(i,len(areas_df))
                centroid = props[areas_df.iloc[i].id].centroid
                closest_index = np.argmin(((correct_centroids - centroid) ** 2).sum(1))
                modified_lobes[L == props[areas_df.iloc[i].id].label] = area_df.iloc[
                    closest_index
                ]["lobe_id"]

    nibabel.save(
        nibabel.Nifti1Image(modified_lobes, img_lobe.affine), f"{output_folder}/{image}"
    )


def lobes_postprocessing():

    try:
        lobes_folder = args.lobe_output
        images = [
            image for image in os.listdir(lobes_folder) if image.endswith(".nii.gz")
        ]

        with multiprocessing.Pool(int(args.num_workers)) as p:
            p.map(convert_lobes, images)
    except Exception as e:
        print("Error in lobes postprocessing: {}".format(e))
        exit()


def convert_infiltration(lung_folder, output_folder, image):


    img_lung = nibabel.load(f"{lung_folder}/{image}")
    arr_lung = img_lung.get_fdata()

    img_infiltration = nibabel.load(f"{output_folder}/{image}")
    arr_infiltration = img_infiltration.get_fdata()

    arr_infiltration = arr_infiltration * (arr_lung > 0)
    nibabel.save(
        nibabel.Nifti1Image(arr_infiltration, img_infiltration.affine),
        f"{output_folder}/{image}",
    )

def convert_ild(lung_folder, output_folder, image):

    img_lung = nibabel.load(f"{lung_folder}/{image}")
    arr_lung = img_lung.get_fdata()

    img_ild = nibabel.load(f"{output_folder}/{image}")
    arr_ild = img_ild.get_fdata()

    arr_ild = arr_ild * (arr_lung > 0)
    nibabel.save(
        nibabel.Nifti1Image(arr_ild, img_ild.affine),
        f"{output_folder}/{image}",
    )

def convert_ggo(lung_folder, output_folder, image):

    img_lung = nibabel.load(f"{lung_folder}/{image}")
    arr_lung = img_lung.get_fdata()

    img_ggo = nibabel.load(f"{output_folder}/{image}")
    arr_ggo = img_ggo.get_fdata()

    arr_ggo = arr_ggo * (arr_lung > 0)
    nibabel.save(
        nibabel.Nifti1Image(arr_ggo, img_ggo.affine),
        f"{output_folder}/{image}",
    )

def convert_ipf(lung_folder, output_folder, image):

    img_lung = nibabel.load(f"{lung_folder}/{image}")
    arr_lung = img_lung.get_fdata()

    img_ipf = nibabel.load(f"{output_folder}/{image}")
    arr_ipf = img_ipf.get_fdata()

    arr_ipf = arr_ipf * (arr_lung > 0)
    nibabel.save(
        nibabel.Nifti1Image(arr_ipf, img_ipf.affine),
        f"{output_folder}/{image}",
    )

def convert_opacities(lung_folder, output_folder, image):
    
        img_lung = nibabel.load(f"{lung_folder}/{image}")
        arr_lung = img_lung.get_fdata()
    
        img_opacities = nibabel.load(f"{output_folder}/{image}")
        arr_opacities = img_opacities.get_fdata()
    
        arr_opacities = arr_opacities * (arr_lung > 0)
        nibabel.save(
            nibabel.Nifti1Image(arr_opacities, img_opacities.affine),
            f"{output_folder}/{image}",
        )

def infiltration_postprocessing():

    try:
        infiltration_folder = args.infiltration_output
        images = [
            image
            for image in os.listdir(infiltration_folder)
            if image.endswith(".nii.gz")
        ]
        with multiprocessing.Pool(int(args.num_workers)) as p:
            p.map(convert_infiltration, images)
    except Exception as e:
        print("Error in infiltration postprocessing: {}".format(e))
        exit()

def get_dummy_dcm(study_id, series_id, rows, cols, position, pixel_spacing=[1, 1], slice_thickness=1, patient_id="123", patient_name="No Name"):

    suffix = '.dcm'
    filename_little_endian = tempfile.NamedTemporaryFile(suffix=suffix).name

    file_meta = pydicom.dataset.FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = '1.2.840.10008.5.1.4.1.1.2'
    file_meta.MediaStorageSOPInstanceUID = "1.2.3"
    file_meta.ImplementationClassUID = "1.2.3.4"
    file_meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian

    dcm = pydicom.FileDataset(filename_or_obj=filename_little_endian, dataset={}, file_meta=file_meta, preamble=b"\0" * 128)
    dcm.StudyInstanceUID = study_id
    dcm.SeriesInstanceUID = series_id
    mod_date = time.strftime("%Y%m%d")
    mod_time = time.strftime("%H%M%S")
    dcm.PatientID = patient_id
    dcm.PatientName = patient_name
    dcm.SamplesPerPixel = 1
    dcm.StudyDate = mod_date
    dcm.StudyTime = mod_time
    dcm.SeriesDate = mod_date
    dcm.SeriesTime = mod_time
    dcm.Modality = 'CT'
    dcm.SOPClassUID = '1.2.840.10008.5.1.4.1.1.2'
    dcm.Rows = rows
    dcm.Columns = cols
    dcm.PixelSpacing = pixel_spacing
    dcm.SliceThickness = slice_thickness
    dcm.BitsAllocated = 16
    dcm.BitsStored = 12
    dcm.HighBit = 11
    dcm.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
    dcm.ImagePositionPatient = position
    dcm.PixelRepresentation = 0
    dcm.SliceLocation = position[2]
    dcm.RescaleIntercept = 0.0
    dcm.RescaleSlope = 1.0
    return dcm


def save_dcm(img, file, data, output_folder, patient_name, max_pixel_val=2):
    sop_id = pydicom.uid.generate_uid()
    img.PhotometricInterpretation = "MONOCHROME1"
    img.LargestImagePixelValue = max_pixel_val
    img[0x0028, 0x0107].VR = "US"
    img.RescaleIntercept = "0.0"
    img.PixelData = data
    img.SOPInstanceUID = sop_id
    # img.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    img.save_as(f"{output_folder}/{file}")

# Nifti to Dicom
def nifti_to_dicom(filename, output_folder, max_pixel_val=None, dicom_path=None):

    patient_name = filename.split("/")[-1].split(".nii")[0]

    if dicom_path is not None:
        
        series = os.listdir(f"{dicom_path}/{patient_name}")[0]
        input_folder = f"{dicom_path}/{patient_name}/{series}"

        dcm = pydicom.dcmread(f"{input_folder}/{os.listdir(input_folder)[0]}")
        nii_image = nibabel.load(f"{filename}").get_fdata()
        shape = nii_image.shape
        axis = np.argmin(shape)

        if axis != 2:
            axes = [0, 1, 2]
            axes.remove(axis)
            nii_image = np.transpose(nii_image, (axes[0], axes[1], axis))

        nii_image = nii_image.astype(dcm.pixel_array.dtype)
        nii_image = np.rot90(nii_image, 1)
        nii_image = nii_image[:, :, ::-1]
        for idx, file in enumerate(
            sorted(os.listdir(input_folder))
        ):  # Read original images's information
            img = pydicom.dcmread(f"{input_folder}/{file}")
            save_dcm(img, file, nii_image[..., idx].tobytes(), output_folder, patient_name, max_pixel_val)
    else:

        nii_image = nibabel.load(f"{filename}")
        arr = nii_image.get_fdata()

        if max_pixel_val is None:
            max_pixel_val = int(np.max(arr))

        affine = nii_image.affine

        image = dicom2nifti.image_volume.ImageVolume(nii_image)

        x_size, y_size, z_size = nii_image.shape
        
        x_pos = affine[0, 3]
        x_mul = affine[0, 0]


        if x_mul > 0:
            x_pos = x_pos - x_mul * (x_size - 1)

        x_pos = -x_pos
            
        y_pos = affine[1, 3]
        y_mul = affine[1, 1]

        if y_mul > 0:
            y_pos = y_pos + y_mul * (y_size - 1)

        y_pos = -y_pos

        z_pos = affine[2, 3]
        z_mul = affine[2, 2]

        if image.coronal_orientation.y_inverted:
            z_pos = z_pos - z_mul * (z_size - 1)
		
        series_id = pydicom.uid.generate_uid()
        study_id = pydicom.uid.generate_uid()
        rows = nii_image.shape[0]
        cols = nii_image.shape[1]
        nii_image = nibabel.load(f"{filename}")
        affine = nii_image.affine
        pixel_spacing = [math.fabs(affine[0, 0]), math.fabs(affine[1, 1])]
        slice_thickness = math.fabs(affine[2, 2])

        arr = arr - arr.min()
        arr = arr.astype('uint16')
        arr = np.flip(arr, axis=1)
        arr = arr.transpose((2, 1, 0))

        for i in range(arr.shape[0]):

            file = f"{i}.dcm"
            position = [x_pos, y_pos, z_pos + i * slice_thickness]
            img = get_dummy_dcm(series_id, study_id, rows, cols, position, pixel_spacing, slice_thickness, patient_id=patient_name, patient_name=patient_name)
            img.InstanceNumber = str(i+1)
            save_dcm(img, file, arr[i, ...].tobytes(), output_folder, patient_name, max_pixel_val)
        

def convert_nifti_to_dicom(nii_folder, dicom_folder, max_pixel_val, dicom_path):

    for file in os.listdir(nii_folder):
        if file.endswith(".nii.gz"):
            nifti_to_dicom(
                f"{nii_folder}/{file}", dicom_folder, max_pixel_val, dicom_path
            )


# Convert labels for lola submission
def convert_lola_labels(input_folder):
    images = [
        f"{input_folder}/{image}"
        for image in os.listdir(input_folder)
        if image.endswith(".nii.gz")
    ]
    for path in images:
        file_name = path.split("/")[1]
        img = nibabel.load(path)
        arr = img.get_fdata()
        arr[arr >= 6] = 0
        arr[arr == 1] = 10
        arr[arr == 2] = 11
        arr[arr == 3] = 20
        arr[arr == 4] = 21
        arr[arr == 5] = 22
        nibabel.save(
            nibabel.Nifti1Image(arr, img.affine), f"{input_folder}/{file_name}"
        )


# Converting Dicom to Nifti
def check_images(nifti_path, dicom_path):

    settings.disable_validate_slice_increment()
    try:

        if dicom_path is None or not os.path.exists(dicom_path):
            print("Images not in dicom format....will check for nifti files")
            sys.stdout.flush()
            if nifti_path is None or not os.path.exists(nifti_path):
                print("Images not in nifti format too....please check the path again")
                sys.stdout.flush()
                return False
            else:
                print("Images already in nifti format")
                sys.stdout.flush()
                for file in os.listdir(nifti_path):
                    if not file.endswith("_0000.nii.gz"):
                        os.system(
                            f"mv {nifti_path}/{file} {nifti_path}/{file.replace('.nii.gz','_0000.nii.gz')}"
                        )
                return True
        else:
            maybe_mkdir(nifti_path)
            count = 0
            print("-" * 20)
            print("Converting dicom to nifti")
            sys.stdout.flush()

            for folder in os.listdir(dicom_path):
                maybe_mkdir(f"{nifti_path}/{folder}")
                dicom2nifti.convert_directory(
                    f"{dicom_path}/{folder}", f"{nifti_path}/{folder}"
                )

            for folder in os.listdir(nifti_path):
                if not folder.endswith(".nii.gz"):
                    try:
                        file = os.listdir(f"{nifti_path}/{folder}")[0]
                    except Exception as e:
                        print(f"Check if this {folder} is right")
                        count += 1
                        os.system(f"rm -r {nifti_path}/{folder}")
                        continue
                    os.system(
                        f"mv {nifti_path}/{folder}/{file} {nifti_path}/{folder}_0000.nii.gz"
                    )

            for f in os.listdir(nifti_path):
                if not f.endswith(".nii.gz"):
                    os.system(f"rm -r {nifti_path}/{f}")
            return True
    except Exception as e:
        print("Error in check_images: {}".format(e))
        return False




def load_data(path, transpose_forward=[0, 1, 2]):
    data = sitk.ReadImage(path)
    properties = {}
    properties["original_size"] = np.array(data.GetSize())[[2, 1, 0]]
    properties["original_spacing"] = np.array(data.GetSpacing())[[2, 1, 0]]

    properties["itk_origin"] = data.GetOrigin()
    properties["itk_spacing"] = data.GetSpacing()
    properties["itk_direction"] = data.GetDirection()

    data = sitk.GetArrayFromImage(data)

    properties["crop_size"] = np.array(data.shape)

    data = np.vstack([data[None]])

    data = data.transpose((0, *[i + 1 for i in transpose_forward]))

    return data, properties


def rescale_data(data, properties, plans, transpose_forward=[0, 1, 2]):

    original_spacing = properties["original_spacing"]
    original_spacing_transposed = np.array(properties["original_spacing"])[
        transpose_forward
    ]
    target_spacing = plans["plans_per_stage"][1]["current_spacing"]
    shape = np.array(data[0].shape)
    new_shape = np.round(
        ((np.array(original_spacing_transposed) / np.array(target_spacing)).astype(float) * shape)
    ).astype(int)

    reshaped = []
    order = 3
    kwargs = {"mode": "edge", "anti_aliasing": False}
    print(
        "Rescaling, original shape: {}, new shape: {}, order: {}".format(
            shape, new_shape, order
        )
    )
    for c in range(data.shape[0]):
        reshaped.append(transform.resize(data[c], new_shape, order, **kwargs)[None])
    reshaped_final_data = np.vstack(reshaped)

    print(reshaped_final_data.shape)

    return reshaped_final_data


def normalize_data(data, plans):
    intensity_proprties = plans["dataset_properties"]["intensityproperties"]
    for c in range(len(data)):
        mean_intensity = intensity_proprties[c]["mean"]
        std_intensity = intensity_proprties[c]["sd"]
        lower_bound = intensity_proprties[c]["percentile_00_5"]
        upper_bound = intensity_proprties[c]["percentile_99_5"]
        data[c] = np.clip(data[c], lower_bound, upper_bound)
        data[c] = (data[c] - mean_intensity) / std_intensity

    return data


## Taken from nnUNet preprocessing code
def preprocess_test_case(data, properties, plans, transpose_forward=[0, 1, 2]):

    try:
        print("Rescaling data")
        data = rescale_data(data, properties, plans, transpose_forward)
        print("Data rescaled")
    except Exception as e:
        print("Error rescaling data")
        print(e)
        sys.exit(-1)

    ### Normalizing data

    try:
        print("Normalizing data")
        data = normalize_data(data, plans)
        print("Data normalized")
    except Exception as e:
        print("Error normalizing data")
        print(e)
        sys.exit(-1)

    return data.astype(np.float32)


def infer(model, data, plans, stage=1, callback=None):

    patch_size = plans["plans_per_stage"][stage]["patch_size"]
    print("Patch size: ", patch_size)
    try:
        print("Predicting")
        output = model._internal_predict_3D_3Dconv_tiled(
            data,
            mirror_axes=(0, 1, 2),
            do_mirroring=False,
            use_gaussian=True,
            all_in_gpu=False,
            step_size=0.5,
            patch_size=patch_size,
            pad_border_mode="constant",
            pad_kwargs={"constant_values": 0},
            regions_class_order=None,
            verbose=True,
            callback=callback
        )
    except Exception as e:
        print("Error predicting")
        print(e)
        sys.exit(-1)

    return output[0]


def inference(args, col, unique_id, model_name, nifti_path, output_path, stage=1):



    print(f"Starting {model_name} inference.....")
    task_name = f"{args[model_name+'_task']}"
    task_fold = f"{args[model_name+'_fold']}"
    plans_path = f"{args['RESULTS_FOLDER']}/nnUNet/3d_fullres/{task_name}/nnUNetTrainerV2__nnUNetPlansv2.1/fold_{task_fold}/model_best.model.pkl"
    weights_path = f"{args['RESULTS_FOLDER']}/nnUNet/3d_fullres/{task_name}/nnUNetTrainerV2__nnUNetPlansv2.1/fold_{task_fold}/model_best.model"
    plans = pickle.load(open(plans_path, 'rb'))['plans']
    model = generic_UNet.Generic_UNet(
        input_channels=1,
        base_num_features=32,
        num_classes=plans['num_classes']+1,
        num_pool=len(plans['plans_per_stage'][1]['pool_op_kernel_sizes']),
        convolutional_pooling=True,
        convolutional_upsampling=True,
        norm_op=torch.nn.InstanceNorm3d,
        conv_op=torch.nn.Conv3d,
        pool_op_kernel_sizes=plans['plans_per_stage'][1]['pool_op_kernel_sizes'],
        conv_kernel_sizes=plans['plans_per_stage'][1]['conv_kernel_sizes'],
        deep_supervision=False
    ).to(args['device'])

    model.load_state_dict(torch.load(weights_path, map_location=args['device'])['state_dict'])
    model.eval()

    col.document(unique_id).update({"predictionStatus": "{} model loaded".format(model_name)})

    data, properties = load_data(nifti_path, transpose_forward=plans['transpose_forward'])

    col.document(unique_id).update({"predictionStatus": "{} data loaded".format(model_name)})

    data = preprocess_test_case(data.astype('float32'), properties, plans, transpose_forward=plans['transpose_forward'])

    col.document(unique_id).update({"predictionStatus": "{} preprocessing done".format(model_name)})

    def callback(x, y, z, steps):
        xi = steps[0].index(x)
        yi = steps[1].index(y)
        zi = steps[2].index(z)
        num_steps_done = xi * len(steps[1]) * len(steps[2]) + yi * len(steps[2]) + zi
        if num_steps_done % 20 == 0:
            col.document(unique_id).update({"predictionStatus": "{} prediction {}/{}".format(model_name, num_steps_done, len(steps[0]) * len(steps[1]) * len(steps[2]))})

    print('data shape   ', data.shape)
    with torch.no_grad():
        output = infer(model, data, plans, stage, callback=callback)

    print('output max', output.max(), output.shape)
    
    col.document(unique_id).update({"predictionStatus": "{} inference done".format(model_name)})


    transpose_forward=plans['transpose_forward']
    # transpose back
    output_transposed = output.transpose([i for i in transpose_forward])

    print('output_transposed max', output_transposed.max(), output_transposed.shape)

    output_resized = resize_segmentation(output_transposed, properties["crop_size"], order=1)


    print('output_resized max', output_resized.max(), output_resized.shape)


    col.document(unique_id).update({"predictionStatus": "{} postprocessing done".format(model_name)})

    output_img = sitk.GetImageFromArray(output_resized)
    output_img.SetSpacing(properties['itk_spacing'])
    output_img.SetOrigin(properties['itk_origin'])
    output_img.SetDirection(properties['itk_direction'])
    sitk.WriteImage(output_img, output_path)


# Main Program starts here

def do_inference(
                config,
                patient_id,
                unique_id,
                models,
                file_type,
                user_id,
            ):

    os.environ["MKL_THREADING_LAYER"] = "GNU"

    """
    Disabling validity dicom checks.
    """
    dicom2nifti.settings.disable_validate_orthogonal()
    dicom2nifti.settings.disable_validate_slice_increment()
    dicom2nifti.settings.disable_validate_instance_number()
    dicom2nifti.settings.disable_validate_slicecount()
    dicom2nifti.settings.disable_validate_orientation()
    dicom2nifti.settings.disable_validate_multiframe_implicit()

    r = redis.Redis(host=config['redis_host'], port=6379)
    firebase_cred = firebase_admin.credentials.Certificate('chestomx-firebase.json')
    firebase_admin.initialize_app(firebase_cred)

    db = firestore.client()
    col = db.collection('predictionRecords')

    labels_colors = config['labels_colors']
    labels = config['labels']


    users_col = db.collection('users')
    ctss_map = users_col.document(user_id).get().to_dict()['ctss']

    niftis_folder = config['NIFTIS_FOLDER']
    outputs_folder = config['OUTPUTS_FOLDER']
    uploads_folder = config['UPLOADS_FOLDER']
    results_folder = config['RESULTS_FOLDER']
    pngs_folder = config['PNGS_FOLDER']
    objs_folder = config['OBJS_FOLDER']


    print('hello', models, outputs_folder, niftis_folder, uploads_folder, pngs_folder)

    # if nifti_path is None:
    #     nifti_path = 'nii_images'
    # if check_images(nifti_path, dicom_path):
    #     print("Finished converting dicom to nifti")
    #     print("-" * 20)
    #     sys.stdout.flush()
    # else:
    #     print("Error in input files")
    #     print("-" * 20)
    #     sys.stdout.flush()
    #     exit()
    # col.document(unique_id).set({
    #     'predictionStatus': 'Inference started',
    #     'patientId': patient_id,
    #     'uniqueId': unique_id,
    #     'models': ','.join(models),
    #     'receivedAt': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    #     'predictedAt': '',
    #     'user_id': user_id
    # })
    # convert to nifti if required
    try:
        nifti_path = f"{niftis_folder}/images/{patient_id}_{unique_id}_0000.nii.gz"
        if file_type == 'dcm':
            dicom_path = f"{uploads_folder}/dcm/{patient_id}_{unique_id}"
            print('dicom_path', dicom_path)
            dcms = [ pydicom.dcmread(f"{dicom_path}/{dcm_file}") for dcm_file in os.listdir(dicom_path) ]
            dicom2nifti.convert_dicom.dicom_array_to_nifti(dcms, nifti_path, True)
            col.document(unique_id).update({'predictionStatus': 'Dicom to nifti conversion done'})
        else:
            os.system(f"cp {uploads_folder}/nii/{patient_id}_{unique_id}.nii.gz {nifti_path}")
            col.document(unique_id).update({'predictionStatus': 'Nifti to nifti conversion done'})
    
    except Exception as e:
        print("Error in dicom to nifti conversion{}".format(e))
        sys.stdout.flush()
        col.document(unique_id).update({'predictionStatus': 'Error in dicom to nifti conversion: {}'.format(e)})
        return
        
    # maybe_mkdir(args.lung_output)
    # maybe_mkdir(args.lobe_output)

    nifti_folder = '/'.join(nifti_path.split('/')[:-1])


    # run models and postprocessing for each model in succession
    try:
        # if 'lungs' in models:
        #     output_path = "{}/lungs/{}_{}.nii.gz".format(outputs_folder, patient_id, unique_id)
        #     inference(args, col, unique_id, "lungs", nifti_path, output_path)
        #     col.document(unique_id).update({'predictionStatus': 'Lungs inference completed!'})
        #     convert_lungs(f"{outputs_folder}/lungs/", f"{outputs_folder}/lungs/", "{}_{}.nii.gz".format(patient_id, unique_id))
        #     col.document(unique_id).update({'predictionStatus': 'Lungs postprocessing completed!'})
        #     # lungs_postprocessing()
        # if 'lobes' in models:
        #     output_path = "{}/lobes/{}_{}.nii.gz".format(outputs_folder, patient_id, unique_id)
        #     inference(args, col, unique_id, "lobes", nifti_path, output_path)
        #     col.document(unique_id).update({'predictionStatus': 'Lobes inference done'})
        #     if 'lungs' in models:
        #         convert_lobes(f"{outputs_folder}/lungs/", f"{outputs_folder}/lobes/", f"{outputs_folder}/lobes/", "{}_{}.nii.gz".format(patient_id, unique_id))
        #         col.document(unique_id).update({'predictionStatus': 'Lobes postprocessing done'})
        #     # if args.convert_lola_labels:
        #     #     convert_lola_labels(args.lobe_output)
        #     #     col.document(unique_id).update({'predictionStatus': 'Lobes conversion done'})
        # if 'infiltration' in models:
        #     output_path = "{}/infiltration/{}_{}.nii.gz".format(outputs_folder, patient_id, unique_id)
        #     inference(args, col, unique_id, "infiltration", nifti_path, output_path)
        #     col.document(unique_id).update({'predictionStatus': 'Infiltration inference done'})
        #     if 'lungs' in models:
        #         convert_infiltration(f"{outputs_folder}/lungs/", f"{outputs_folder}/infiltration/", "{}_{}.nii.gz".format(patient_id, unique_id))
        #         col.document(unique_id).update({'predictionStatus': 'Infiltration postprocessing done'})
        # if 'ild' in models:
        #     output_path = "{}/ild/{}_{}.nii.gz".format(outputs_folder, patient_id, unique_id)
        #     inference(args, col, unique_id, "ild", nifti_path, output_path)
        #     col.document(unique_id).update({'predictionStatus': 'ILD inference done'})
        #     if 'lungs' in models:
        #         convert_ild(f"{outputs_folder}/lungs/", f"{outputs_folder}/ild/", "{}_{}.nii.gz".format(patient_id, unique_id))
        #         col.document(unique_id).update({'predictionStatus': 'ILD postprocessing done'})
        for model in models:
            output_path = "{}/{}/{}_{}.nii.gz".format(outputs_folder, model, patient_id, unique_id)
            inference(config, col, unique_id, model, nifti_path, output_path)
            col.document(unique_id).update({'predictionStatus': '{} inference done'.format(model)})
            # postprocessing possible only if lungs model is also chosen
            if 'lungs' in models:
                postprocessing[model](f"{outputs_folder}/lungs/", f"{outputs_folder}/{model}/", "{}_{}.nii.gz".format(patient_id, unique_id))
                col.document(unique_id).update({'predictionStatus': '{} postprocessing done'.format(model)})

            # if args.convert_lola_labels:
            #     convert_lola_labels(args.lobe_output)
            #     col.document(unique_id).update({'predictionStatus': 'Lobes conversion done'})

        # we can calculate ctss if both lobes and infiltration models are chosen
        if 'lobes' in models and 'infiltration' in models:
            lobes_path = f"{outputs_folder}/lobes/{patient_id}_{unique_id}.nii.gz"
            infiltration_path = f"{outputs_folder}/infiltration/{patient_id}_{unique_id}.nii.gz"
            involements, ctss = calc_ctss(lobes_path, infiltration_path, ctss_map)
            ctss = ctss_results_map(involements, ctss)
            col.document(unique_id).update({
                'ctss': ctss,
            })
            col.document(unique_id).update({'predictionStatus': 'CTSS calculation done'})
    except Exception as e:
        print(
            "Error in inference...check if fold no given properly or if the task is correct {}".format(
                e
            )
        )
        sys.stdout.flush()
        col.document(unique_id).update({'predictionStatus': 'Error in inference: {}'.format(e)})
        return


    print("Finished inferencing")
    print("-" * 20)

    col.document(unique_id).update({'predictionStatus': 'Finished inference'})

    access_key = uuid.uuid4().hex


    # make png of the results
    utils.make_pngs_of_lobes(patient_id, unique_id, niftis_folder=niftis_folder, outputs_folder=outputs_folder, pngs_folder=pngs_folder, lola=False, models=models)

    col.document(unique_id).update({'predictionStatus': 'Finished png generation'})

    # convert each label to obj file for 3d rendering
    for model in models:
        nii_image_path = f"{outputs_folder}/{model}/{patient_id}_{unique_id}.nii.gz"
        obj_file_path = f"{objs_folder}/{model}/{patient_id}_{unique_id}.obj"
        # convert_to_obj.convert(nii_image_path, obj_file_path, labels[model])
        convert_to_obj2.convert(nii_image_path, obj_file_path, ['background'] + list(labels[model].keys()))

    col.document(unique_id).update({'predictionStatus': 'Finished obj generation'})

    # convert each label to dicoms
    for model in models:
        nii_image_path = f"{outputs_folder}/{model}/{patient_id}_{unique_id}.nii.gz"
        dicoms_zip_path = f"{outputs_folder}/{model}/{patient_id}_{unique_id}.zip"
        maybe_mkdir("./temp_dicoms")
        nifti_to_dicom(nii_image_path, "./temp_dicoms")
        os.system(f"zip -r {dicoms_zip_path} temp_dicoms/")
        os.system(f"rm -r temp_dicoms")

    make_reports.make_report(
        patient_id=patient_id,
        unique_id=unique_id,
        models=models,
        model_labels=config['labels'],
        ignore=config['ignore'],
        niftis_folder=niftis_folder,
        uploads_folder=uploads_folder,
        outputs_folder=outputs_folder,
        templates_folder=config['TEMPLATES_FOLDER'],
        reports_folder=config['REPORTS_FOLDER'],
    )

    col.document(unique_id).update({'predictionStatus': 'Finished report generation'})


    col.document(unique_id).update({
        'predictionStatus': 'Finished Processing!', 
        'predictedAt': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'accessKey': access_key
    })


    r.set(unique_id, access_key)

    # send mail after everything done
    try:
        mail_to = db.collection('users').document(user_id).get().to_dict()['email']

        if mail_to is not None:
            utils.send_prediction_mail(patient_id, unique_id, access_key, mail_to, config)
            
    except Exception as e:
        print(e)
        print("Could not send mail", str(e))

    return

def calc_metrics(unique_id, model, pred_file_path, label_file_path, model_labels, mail_to, config):
    try:

        firebase_cred = firebase_admin.credentials.Certificate('chestomx-firebase.json')
        firebase_admin.initialize_app(firebase_cred)

        db = firestore.client()
        col = db.collection('predictionRecords')

        label_img = nibabel.load(label_file_path).get_fdata()
        pred_img = nibabel.load(pred_file_path).get_fdata()

        metrics = []
 
        # go through each label in the model and get the metrics
        idx = 1
        for model_label in model_labels:
            metrics.append({model_label: get_metrics(label_img == idx, pred_img == idx)})
            idx += 1
        col.document(unique_id).update({
            'metrics.{}'.format(model): metrics
        })

        # send the metrics to the corresponding user mail
        if mail_to is not None:
            utils.send_metrics_mail(unique_id, metrics, model, model_labels, mail_to, config)
    except Exception as e:
        print('error in calculating metrics: ', str(e))
        col.document(unique_id).update({
            'metrics': {
                model: 'Error in calculating metrics: ' + str(e)
            }
        })

if __name__ == '__main__':
    # patient_id, unique_id = 'lola11_7_1ee0f8ea383948b2bff2cc22241d0020'.split('_')
    patient_id = '01062933'
    unique_id = 'b480c4ca652a44c8a1d401e2c103de43'
    file_type = 'dcm'
    config = yaml.load(open('config.yaml'), Loader=yaml.Loader)
    args = {
            'redis_host': config['redis_host'],
            'patient_id': patient_id,
            'unique_id': unique_id,
            'file_type': file_type,
            'models': [ 'lungs' ],
            'device': config['device'],
            'lungs_fold': config['lungs_fold'],
            'lobes_fold': config['lobes_fold'],
            'infiltration_fold': config['infiltration_fold'],
            'lungs_task': config['lungs_task'],
            'lobes_task': config['lobes_task'],
            'infiltration_task': config['infiltration_task'],
            'RESULTS_FOLDER': config['RESULTS_FOLDER'],
        }
    do_inference(args)

postprocessing = {
    'lungs': convert_lungs,
    'infiltration': convert_infiltration,
    'fibrosis_7labels': convert_ild,
    'lobes': convert_lobes,
    'ggo_consolidation': convert_ggo,
    'fibrosis_4labels': convert_ipf,
    'haa': convert_opacities
}
