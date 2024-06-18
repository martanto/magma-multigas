#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import xml.etree.ElementTree as ET

tree = ET.parse('metadata_v4.xml')
root = tree.getroot()

detailed_six_hours = root[2][0]

columns = []

for index in range(0, len(detailed_six_hours)):
    attributes = detailed_six_hours[index]
    tag = attributes.tag
    attr_dict = {}
    if tag == 'attr':
        attr_dict['name'] = attributes[0].text
        attr_dict['description'] = attributes[1].text

        # for attrdomv in attributes.iter('udom'):
        #     print(attrdomv.text)
        columns.append(attr_dict)

columns_description: pd.DataFrame = pd.DataFrame(columns)
