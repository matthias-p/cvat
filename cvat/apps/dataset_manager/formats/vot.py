# Copyright (C) 2021 Intel Corporation
#
# SPDX-License-Identifier: MIT

from tempfile import TemporaryDirectory
import zipfile
import io
from pathlib import Path

import cv2
from .registry import importer, exporter
from cvat.apps.dataset_manager.bindings import CvatTaskDataExtractor, TaskData
from datumaro.components.dataset import DatasetItem


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
    attribute_copy = attributes.copy()
    for key, values in attributes.items():
        for value in values:
            if value == 1 and key in attribute_mapping.keys():
                break
        else:
            attribute_copy.pop(key)
            continue
    return attribute_copy


@importer(name="VOT", version="1.0", ext="ZIP")
def vot_importer(file_object, task_data: TaskData, **options):
    if zipfile.is_zipfile(file_object):
        with TemporaryDirectory() as temp_dir:
            zipfile.ZipFile(file_object).extractall(temp_dir)
            p = Path(temp_dir)
            gt_file_path = p / "groundtruth.txt"
            att_files = list(p.glob("*.tag"))

            groundtruths = []
            attributes = {}

            with open(gt_file_path, "r") as gt_file:
                groundtruths = gt_file.read().rstrip("\n").splitlines()

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
                    attributes=[task_data.Attribute(name=name, value=True if value == "1" else False) for name, value in attribute_list],
                    label="object",
                    source="manual",
                    frame=x,
                )

                task_data.add_shape(shape)



@exporter(name="VOT", version="1.0", ext="ZIP")
def vot_exporter(file_object, task_data: TaskData, save_images=False):
    extractor = CvatTaskDataExtractor(task_data=task_data, include_images=save_images)
    task_meta = task_data.meta.get("task")

    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "a", zipfile.ZIP_DEFLATED, False) as zip_file:
        attributes = {}
        groundtruths = []
        dataset_item: DatasetItem
        for dataset_item in extractor._items:
            if save_images and dataset_item.has_image:
                if dataset_item.image.has_data and dataset_item.image.has_size:
                    _, data = cv2.imencode(".png", dataset_item.image.data)
                    zip_file.writestr("data/" + dataset_item.id + ".png", data)
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
        attributes_files = []
        for key, values in attributes.items():
            att_buffer = io.StringIO()
            for value in values:
                att_buffer.write(f"{value}\n")
            attributes_files.append((f"{key}.tag", att_buffer))
        for file_name, data in attributes_files:
            zip_file.writestr(file_name, data.getvalue())

        # write info.txt file
        info = _make_info(task_meta)
        info["attributes"] = ",".join(_map_attributes(attributes))
        meta_buffer = io.StringIO()
        for key, value in info.items():
            meta_buffer.write(f"{key}={value}\n")
        zip_file.writestr("info.txt", meta_buffer.getvalue())

        # write gt file
        gt_buffer = io.StringIO()
        for line in groundtruths:
            gt_buffer.write(f"{line}\n")
        zip_file.writestr("groundtruth.txt", gt_buffer.getvalue())

        # write sequence file
        seq_buffer = io.StringIO()
        seq_buffer.write("channels.{spectrum}=data/%08d.png\n")
        seq_buffer.write(f"start_frame={info.get('start_frame')}\n")
        seq_buffer.write("format=default\n")
        seq_buffer.write("fps={fps}\n")
        seq_buffer.write("name={sequence_name}")
        zip_file.writestr("sequence", seq_buffer.getvalue())

    file_object.write(zip_buffer.getvalue())

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
