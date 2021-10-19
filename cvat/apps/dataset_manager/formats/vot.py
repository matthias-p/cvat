# Copyright (C) 2021 Intel Corporation
#
# SPDX-License-Identifier: MIT

import zipfile
from pathlib import Path
from tempfile import TemporaryDirectory

import cv2
from datumaro.components.dataset import DatasetItem

from cvat.apps.dataset_manager.bindings import CvatTaskDataExtractor, TaskData
from cvat.apps.dataset_manager.util import make_zip_archive
from .registry import importer, exporter

attribute_mapping = {
    "appearance_change": "AC",
    "aspect_ratio_change": "ARC",
    "background_clutter": "BC",
    "camera_motion": "CM",
    "fast_motion": "FM",
    "full_occlusion": "FOC",
    "illumination_variation": "IV",
    "low_resolution": "LR",
    "motion_blur": "MB",
    "out_of_view": "OV",
    "partial_occlusion": "POC",
    "scale_variation": "SV",
    "similar_objects": "SO",
}


def _map_attributes(attributes: dict):
    return [attribute_mapping.get(key) for key in attributes.keys()]


def _make_info(task_meta: dict):
    """
    Create dictionary with metainfo for the sequence
    """
    meta_info = {}
    meta_info["sequence_name"] = task_meta.get("name")
    meta_info["spectrum"] = "color"
    meta_info["camera_name"] = "n/a"
    meta_info["frame_size_x"] = 0
    meta_info["frame_size_y"] = 0
    meta_info["number_of_frames"] = task_meta.get("size")
    meta_info["attributes"] = ""
    meta_info["location"] = "n/a"
    meta_info["year"] = 0
    meta_info["original_video_file"] = "n/a"
    meta_info["original_frame_size_x"] = 0
    meta_info["original_frame_size_y"] = 0
    meta_info["cropping_offset_x"] = 0
    meta_info["cropping_offset_y"] = 0
    meta_info["deinterlaced"] = 0
    meta_info["start_frame"] = int(task_meta.get("start_frame")) + 1
    meta_info["end_frame"] = int(task_meta.get("stop_frame")) + 1
    return meta_info


def _clear_attributes(attributes: dict):
    """
    Remove attributes which are False permanently and are not part of the VOT attributes
    """
    attribute_copy = attributes.copy()
    for key, values in attributes.items():
        for value in values:
            if value == 1 and (key in attribute_mapping.keys() or key in attribute_mapping.items()):
                break
        else:
            attribute_copy.pop(key)
            continue
    return attribute_copy


@importer(name="VOT", version="1.0", ext="ZIP")
def vot_importer(file_object, task_data: TaskData):
    if zipfile.is_zipfile(file_object):
        with TemporaryDirectory() as temp_dir:
            zipfile.ZipFile(file_object).extractall(temp_dir)
            p = Path(temp_dir)
            gt_file_path = p / "groundtruth.txt"
            att_files = list(p.glob("*.tag"))

            attributes = {}

            assert gt_file_path.exists() and gt_file_path.is_file()
            with open(gt_file_path, "r") as gt_file:
                groundtruths = gt_file.read().rstrip("\n").splitlines()

            if att_files is not None:
                for att_file in att_files:
                    with open(att_file, "r") as att_f:
                        attributes[att_file.stem] = att_f.read().rstrip("\n").splitlines()

            for x, line in enumerate(groundtruths):
                if "nan" in line:
                    line = [0, 0, 0, 0, 0, 0, 0, 0]
                else:
                    line = [float(val) for val in line.split(",")]

                attribute_list = []
                for key, value in attributes.items():
                    attribute_list.append((key, value[x]))

                shape = task_data.LabeledShape(
                    type="rectangle",
                    points=[line[0], line[1], line[4], line[5]],
                    occluded=False,
                    attributes=[task_data.Attribute(name=name, value=True if value == "1" else False)
                                for name, value in attribute_list],
                    label="object",
                    source="manual",
                    frame=x,
                )

                task_data.add_shape(shape)


@exporter(name="VOT", version="1.0", ext="ZIP")
def vot_exporter(file_object, task_data: TaskData, save_images=False):
    extractor = CvatTaskDataExtractor(task_data=task_data, include_images=save_images)
    task_meta = task_data.meta.get("task")

    with TemporaryDirectory() as tempdir:
        tempdir_path = Path(tempdir)
        if save_images:
            image_path = tempdir_path / "data"
            image_path.mkdir(exist_ok=True)

        attributes = {}
        groundtruths = []
        dataset_item: DatasetItem
        for x, dataset_item in enumerate(extractor._items, start=1):
            if save_images and dataset_item.has_image:
                if dataset_item.image.has_data and dataset_item.image.has_size:
                    _, data = cv2.imencode(".png", dataset_item.image.data)
                    with open(image_path / f"{x}.png", "wb") as img_file:
                        img_file.write(data)
            for annotation in dataset_item.annotations:
                xtl = annotation.points[0]
                ytl = annotation.points[1]
                xbr = annotation.points[2]
                ybr = annotation.points[3]

                for key, value in annotation.attributes.items():
                    if not attributes.get(key):
                        attributes[key] = []
                    attributes[key].append(1 if value else 0)

                if attributes.get("full_occlusion")[-1] == 1 or attributes.get("out_of_view")[-1] == 1:
                    groundtruths.append("nan,nan,nan,nan,nan,nan,nan,nan")
                else:
                    groundtruths.append(f"{xtl},{ytl},{xtl},{ybr},{xbr},{ybr},{xbr},{ytl}")

        # write .tag files
        attributes = _clear_attributes(attributes)
        for key, values in attributes.items():
            with open(tempdir_path / f"{key}.tag", "w") as att_file:
                for value in values:
                    att_file.write(f"{value}\n")

        # write gt file
        with open(tempdir_path / "groundtruth.txt", "w") as gt_file:
            for line in groundtruths:
                gt_file.write(f"{line}\n")

        # Those files are normally not written in the VOT challenge but may help other people with their tasks
        # write info.txt file
        info = _make_info(task_meta)
        info["attributes"] = ",".join(_map_attributes(attributes))
        with open(tempdir_path / "info.txt", "w") as info_file:
            for key, value in info.items():
                info_file.write(f"{key}={value}\n")

        # write sequence file
        with open(tempdir_path / "sequence", "w") as seq_file:
            seq_file.write("channels.{spectrum}=data/%08d.png\n")
            seq_file.write(f"start_frame={info.get('start_frame')}\n")
            seq_file.write("format=default\n")
            seq_file.write("fps={fps}\n")
            seq_file.write("name={sequence_name}")

        make_zip_archive(tempdir, file_object)


"""
Example for attributes which are used in VOT
[
    {
        "name": "object",
        "color": "#40e020",
        "attributes": [
            {
                "name": "appearance_change",
                "input_type": "checkbox",
                "mutable": true,
                "values": [
                    "false"
                ]
            },
            {
                "name": "aspect_ratio_change",
                "input_type": "checkbox",
                "mutable": true,
                "values": [
                    "false"
                ]
            },
            {
                "name": "background_clutter",
                "input_type": "checkbox",
                "mutable": false,
                "values": [
                    "false"
                ]
            },
            {
                "name": "camera_motion",
                "input_type": "checkbox",
                "mutable": false,
                "values": [
                    "false"
                ]
            },
            {
                "name": "fast_motion",
                "input_type": "checkbox",
                "mutable": true,
                "values": [
                    "false"
                ]
            },
            {
                "name": "full_occlusion",
                "input_type": "checkbox",
                "mutable": true,
                "values": [
                    "false"
                ]
            },
            {
                "name": "illumination_variation",
                "input_type": "checkbox",
                "mutable": false,
                "values": [
                    "false"
                ]
            },
            {
                "name": "low_resolution",
                "input_type": "checkbox",
                "mutable": false,
                "values": [
                    "false"
                ]
            },
            {
                "name": "motion_blur",
                "input_type": "checkbox",
                "mutable": false,
                "values": [
                    "false"
                ]
            },
            {
                "name": "out_of_view",
                "input_type": "checkbox",
                "mutable": true,
                "values": [
                    "false"
                ]
            },
            {
                "name": "partial_occlusion",
                "input_type": "checkbox",
                "mutable": true,
                "values": [
                    "false"
                ]
            },
            {
                "name": "scale_variation",
                "input_type": "checkbox",
                "mutable": true,
                "values": [
                    "false"
                ]
            },
            {
                "name": "similar_objects",
                "input_type": "checkbox",
                "mutable": false,
                "values": [
                    "false"
                ]
            }
        ]
    }
]
"""
