#!/usr/bin/env pytest
# -*- coding: utf-8 -*-
###############################################################################
#
# Project:  GDAL/OGR Test Suite
# Purpose:  Benchmarking of gti
# Author:   Even Rouault <even dot rouault at spatialys.com>
#
###############################################################################
# Copyright (c) 2023, Even Rouault <even dot rouault at spatialys.com>
#
# SPDX-License-Identifier: MIT
###############################################################################

import gdaltest
import pytest

from osgeo import gdal, ogr

import numpy as np


def test_composite_src_with_mask_into_dest(benchmark):
    ds = gdal.GetDriverByName("GTiff").Create('/vsimem/one.tif', 1000, 1000, 2)
    ds.SetGeoTransform([0, 1, 0, 0, 0, -1])
    ds.GetRasterBand(2).SetColorInterpretation(gdal.GCI_AlphaBand)

    arr = np.random.default_rng().choice([0x00, 0xFF], 1000 * 1000)
    ds.GetRasterBand(2).WriteArray(arr.astype(np.uint8).reshape(1000, 1000))

    del ds

    index_ds = ogr.GetDriverByName("GPKG").CreateDataSource(
        "/vsimem/index.gti.gpkg")
    lyr = index_ds.CreateLayer(
        "index", srs=(gdal.Open("/vsimem/one.tif").GetSpatialRef()))
    lyr.CreateField(ogr.FieldDefn("location"))

    for x in ['/vsimem/one.tif', '/vsimem/one.tif']:
        dsf = gdal.Open(x)
        f = ogr.Feature(lyr.GetLayerDefn())
        src_gt = dsf.GetGeoTransform()
        minx = src_gt[0]
        maxx = minx + dsf.RasterXSize * src_gt[1]
        maxy = src_gt[3]
        miny = maxy + dsf.RasterYSize * src_gt[5]
        f['location'] = dsf.GetDescription()
        f.SetGeometry(
            ogr.CreateGeometryFromWkt(
                f"POLYGON(({minx} {miny},{minx} {maxy},{maxx} {maxy},{maxx} {miny},{minx} {miny}))"
            ))
        lyr.CreateFeature(f)

    del index_ds

    vrt_ds = gdal.Open("/vsimem/index.gti.gpkg")
    b1 = vrt_ds.GetRasterBand(1)

    benchmark(lambda: [b1.ReadRaster() for _ in range(100)])
