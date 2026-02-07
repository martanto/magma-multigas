#!/usr/bin/env python
# -*- coding: utf-8 -*-

from xml.etree import ElementTree

import pandas as pd
from importlib_resources import as_file, files

xml_path = files("magma_multigas.resources").joinpath("metadata_v4.xml")

with as_file(xml_path) as xml_file:
    tree = ElementTree.parse(xml_file)
    root = tree.getroot()

detailed_six_hours = root[2][0]

columns = []

for index in range(0, len(detailed_six_hours)):
    attributes = detailed_six_hours[index]
    tag = attributes.tag
    attr_dict = {}
    if tag == "attr":
        attr_dict["name"] = attributes[0].text
        attr_dict["description"] = attributes[1].text

        # for attrdomv in attributes.iter('udom'):
        #     print(attrdomv.text)
        columns.append(attr_dict)

columns_description: pd.DataFrame = pd.DataFrame(columns)
