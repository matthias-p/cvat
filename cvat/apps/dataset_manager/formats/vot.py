# Copyright (C) 2021 Intel Corporation
#
# SPDX-License-Identifier: MIT

from .registry import importer, exporter


@importer(name="VOT", version="1.0", ext="ZIP")
def vot_importer(file_object, task_data, **options):
    pass


@exporter(name="VOT", version="1.0", ext="ZIP")
def vot_exporter(file_object, task_data, **options):
    pass


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
