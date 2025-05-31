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

from osgeo import gdal


@pytest.mark.parametrize("data_type", [gdal.GDT_Byte, gdal.GDT_UInt16])
@pytest.mark.parametrize("nbands", [3, 4])
def test_vrtpansharpen(benchmark, data_type, nbands):

    xs = 4000
    ys = 4000

    ds = gdal.GetDriverByName("GTiff").Create("/vsimem/pan.tif", xs, ys, 1,
                                              data_type)
    ds.SetGeoTransform([0, 1, 0, 0, 0, -1])
    ds = None

    vrt = """
    <VRTDataset subClass="VRTPansharpenedDataset">
    <PansharpeningOptions>
        <Algorithm>WeightedBrovey</Algorithm>
        <PanchroBand>
                <SourceFilename>/vsimem/pan.tif</SourceFilename>
                <SourceBand>1</SourceBand>
        </PanchroBand>
        """
    for i in range(nbands):
        vrt += f"""
        <SpectralBand dstBand="{i + 1}">
                <SourceFilename>/vsimem/pan.tif</SourceFilename>
                <SourceBand>1</SourceBand>
        </SpectralBand>
        """
    vrt += """
    </PansharpeningOptions>
    </VRTDataset>
    """

    vrt_ds = gdal.Open(vrt)

    # pansharpen evaluates lazily in IRasterIO
    with gdaltest.config_option("GDAL_NUM_THREADS", "1"):
        res_ds = benchmark(gdal.GetDriverByName("MEM").CreateCopy, '', vrt_ds)

    gdal.Unlink("/vsimem/pan.tif")

    assert res_ds != None
