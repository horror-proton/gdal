#!/usr/bin/env pytest
# -*- coding: utf-8 -*-
###############################################################################
#
# Project:  GDAL/OGR Test Suite
# Purpose:  Benchmarking of gdaldither
# Author:   Even Rouault <even dot rouault at spatialys.com>
#
###############################################################################
# Copyright (c) 2023, Even Rouault <even dot rouault at spatialys.com>
#
# SPDX-License-Identifier: MIT
###############################################################################

import gdaltest
import pytest

from osgeo import gdal

import numpy as np


# TODO: find test data from test_pct.py?
def test_rgb2pct(benchmark):
    ds = gdal.GetDriverByName("MEM").Create('', 100, 100, 3, gdal.GDT_Byte)

    rng = np.random.default_rng()
    r_band = ds.GetRasterBand(1)
    g_band = ds.GetRasterBand(2)
    b_band = ds.GetRasterBand(3)

    r_band.WriteRaster(0, 0, 100, 100, rng.bytes(100 * 100))
    g_band.WriteRaster(0, 0, 100, 100, rng.bytes(100 * 100))
    b_band.WriteRaster(0, 0, 100, 100, rng.bytes(100 * 100))

    ct = gdal.ColorTable()

    gdal.ComputeMedianCutPCT(r_band, g_band, b_band, 255, ct)

    dst_ds = gdal.GetDriverByName("MEM").Create('', 100, 100, 1, gdal.GDT_Byte)
    dst_band = dst_ds.GetRasterBand(1)

    benchmark(lambda: [
        gdal.DitherRGB2PCT(r_band, g_band, b_band, dst_band, ct)
        for _ in range(100)
    ])
