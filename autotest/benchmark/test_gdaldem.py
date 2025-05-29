#!/usr/bin/env pytest
# -*- coding: utf-8 -*-
###############################################################################
#
# Project:  GDAL/OGR Test Suite
# Purpose:  Benchmarking of gdalwarp
# Author:   Even Rouault <even dot rouault at spatialys.com>
#
###############################################################################
# Copyright (c) 2023, Even Rouault <even dot rouault at spatialys.com>
#
# SPDX-License-Identifier: MIT
###############################################################################

import gdaltest
import pytest

from osgeo import gdal, osr

# Must be set to run the test_XXX functions under the benchmark fixture
pytestmark = pytest.mark.usefixtures("decorate_with_benchmark")


def test_gdaldem_hillshade():
    src_ds = gdal.GetDriverByName("MEM").Create("", 10000, 10000, 1)
    ds = gdal.DEMProcessing("", src_ds, "hillshade", format="MEM")
    return ds
