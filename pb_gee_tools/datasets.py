#!/usr/bin/env python

# This file is part of 'pb_gee_tools'
#
# Copyright 2024 Pete Bunting
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
#
# Purpose:  Get access to datasets.
#
# Author: Pete Bunting
# Email: pfb@aber.ac.uk
# Date: 17/07/2025
# Version: 1.0
#
# History:
# Version 1.0 - Created.

import ee
import datetime
import pb_gee_tools.utils


def get_sr_landsat_collection(
    aoi: ee.Geometry,
    start_date: datetime.datetime,
    end_date: datetime.datetime,
    cloud_thres: int = 50,
    ignore_ls7: bool = False,
    out_lstm_bands: bool = True,
) -> ee.ImageCollection:
    """
    A function which returns an GEE Image Collection
    :param aoi: an ee.Geometry object representing the area of interest
    :param start_date: the start date for the collection
    :param end_date: the end date for the collection
    :param cloud_thres: a cloud threshold for the scenes to be included
    :param ignore_ls7: A boolean specifying whether to ignore landsat 7 should be
                       ignored which might be preferable to the SLC-off error.
                       Default: False
    :param out_lstm_bands: A boolean specifying whether to output the LS8 and LS9
                           outputs should be subset to remove the coastal band so
                           the bands are compatible with LS7 and LS5/4.
    :return: A GEE Image Collection of the landsat images.

    """
    ls_start = datetime.datetime(year=1982, month=7, day=16)
    ls_end = datetime.datetime.now()

    if not pb_gee_tools.utils.do_dates_overlap(
        s1_date=start_date, e1_date=end_date, s2_date=ls_start, e2_date=ls_end
    ):
        raise Exception(
            "Date range specified does not overlap "
            "with the availability of Landsat imagery."
        )

    ee_start_date = ee.Date.fromYMD(
        ee.Number(start_date.year),
        ee.Number(start_date.month),
        ee.Number(start_date.day),
    )
    ee_end_date = ee.Date.fromYMD(
        ee.Number(end_date.year), ee.Number(end_date.month), ee.Number(end_date.day)
    )

    def _read_ls_col(ls_col, aoi, start_date: ee.Date, end_date: ee.Date, cloud_thres):
        return (
            ls_col.filterBounds(aoi)
            .filterDate(start_date, end_date)
            .filter(f"CLOUD_COVER < {cloud_thres}")
        )

    def _mask_clouds(img):
        qa_mask = img.select("QA_PIXEL").bitwiseAnd(int("11111", 2)).eq(0)
        sat_mask = img.select("QA_RADSAT").eq(0)
        return img.updateMask(qa_mask).updateMask(sat_mask)

    def _oli_band_tm_sel_rename(img):
        return img.select(
            ["SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B6", "SR_B7"],
            ["Blue", "Green", "Red", "NIR", "SWIR1", "SWIR2"],
        )

    def _oli_band_sel_rename(img):
        return img.select(
            ["SR_B1", "SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B6", "SR_B7"],
            ["Coastal", "Blue", "Green", "Red", "NIR", "SWIR1", "SWIR2"],
        )

    def _etm_band_sel_rename(img):
        return img.select(
            ["SR_B1", "SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B7"],
            ["Blue", "Green", "Red", "NIR", "SWIR1", "SWIR2"],
        )

    def _tm_band_sel_rename(img):
        return img.select(
            ["SR_B1", "SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B7"],
            ["Blue", "Green", "Red", "NIR", "SWIR1", "SWIR2"],
        )

    def apply_scale_factors(image):
        optical_bands = (
            image.select("SR_B.").multiply(0.0000275).add(-0.2).multiply(10000)
        )
        return image.addBands(optical_bands, None, True)

    lc09_col = ee.ImageCollection("LANDSAT/LC09/C02/T1_L2")
    lc08_col = ee.ImageCollection("LANDSAT/LC08/C02/T1_L2")
    le07_col = ee.ImageCollection("LANDSAT/LE07/C02/T1_L2")
    lt05_col = ee.ImageCollection("LANDSAT/LT05/C02/T1_L2")
    lt04_col = ee.ImageCollection("LANDSAT/LT04/C02/T1_L2")

    ls4_start = datetime.datetime(year=1982, month=8, day=1)
    ls4_end = datetime.datetime(year=1993, month=12, day=31)

    ls5_start = datetime.datetime(year=1984, month=3, day=1)
    ls5_end = datetime.datetime(year=2013, month=6, day=1)

    ls7_start = datetime.datetime(year=1999, month=3, day=1)
    ls7_end = datetime.datetime(year=2022, month=3, day=1)

    ls8_start = datetime.datetime(year=2013, month=3, day=1)
    ls8_end = datetime.datetime.now()

    ls9_start = datetime.datetime(year=2021, month=11, day=1)
    ls9_end = datetime.datetime.now()

    use_ls9 = False
    use_ls8 = False
    use_ls7 = False
    use_ls5 = False
    use_ls4 = False
    if pb_gee_tools.utils.do_dates_overlap(
        s1_date=start_date, e1_date=end_date, s2_date=ls9_start, e2_date=ls9_end
    ):
        use_ls9 = True
    if pb_gee_tools.utils.do_dates_overlap(
        s1_date=start_date, e1_date=end_date, s2_date=ls8_start, e2_date=ls8_end
    ):
        use_ls8 = True
    if pb_gee_tools.utils.do_dates_overlap(
        s1_date=start_date, e1_date=end_date, s2_date=ls7_start, e2_date=ls7_end
    ):
        use_ls7 = True
    if pb_gee_tools.utils.do_dates_overlap(
        s1_date=start_date, e1_date=end_date, s2_date=ls5_start, e2_date=ls5_end
    ):
        use_ls5 = True
    if pb_gee_tools.utils.do_dates_overlap(
        s1_date=start_date, e1_date=end_date, s2_date=ls4_start, e2_date=ls4_end
    ):
        use_ls4 = True
    if ignore_ls7:
        use_ls7 = False

    if use_ls4:
        ls4_img_col = _read_ls_col(
            lt04_col, aoi, ee_start_date, ee_end_date, cloud_thres
        )
        ls4_img_col = ls4_img_col.map(apply_scale_factors)

        ls4_img_col = ls4_img_col.map(_mask_clouds).map(_tm_band_sel_rename)
        out_col = ls4_img_col
    if use_ls5:
        ls5_img_col = _read_ls_col(
            lt05_col, aoi, ee_start_date, ee_end_date, cloud_thres
        )
        ls5_img_col = ls5_img_col.map(apply_scale_factors)

        ls5_img_col = ls5_img_col.map(_mask_clouds).map(_tm_band_sel_rename)
        if not use_ls4:
            out_col = ls5_img_col
        else:
            out_col = out_col.merge(ls5_img_col)
    if use_ls7:
        ls7_img_col = _read_ls_col(
            le07_col, aoi, ee_start_date, ee_end_date, cloud_thres
        )
        ls7_img_col = ls7_img_col.map(apply_scale_factors)

        ls7_img_col = ls7_img_col.map(_mask_clouds).map(_etm_band_sel_rename)
        if (not use_ls4) and (not use_ls5):
            out_col = ls7_img_col
        else:
            out_col = out_col.merge(ls7_img_col)
    if use_ls8:
        ls8_img_col = _read_ls_col(
            lc08_col, aoi, ee_start_date, ee_end_date, cloud_thres
        )
        ls8_img_col = ls8_img_col.map(apply_scale_factors)

        if out_lstm_bands:
            ls8_img_col = ls8_img_col.map(_mask_clouds).map(_oli_band_tm_sel_rename)
        else:
            ls8_img_col = ls8_img_col.map(_mask_clouds).map(_oli_band_sel_rename)
        if (not use_ls4) and (not use_ls5) and (not use_ls7):
            out_col = ls8_img_col
        else:
            out_col = out_col.merge(ls8_img_col)
    if use_ls9:
        ls9_img_col = _read_ls_col(
            lc09_col, aoi, ee_start_date, ee_end_date, cloud_thres
        )
        ls9_img_col = ls9_img_col.map(apply_scale_factors)

        if out_lstm_bands:
            ls9_img_col = ls9_img_col.map(_mask_clouds).map(_oli_band_tm_sel_rename)
        else:
            ls9_img_col = ls9_img_col.map(_mask_clouds).map(_oli_band_sel_rename)
        if (not use_ls4) and (not use_ls5) and (not use_ls7) and (not use_ls8):
            out_col = ls9_img_col
        else:
            out_col = out_col.merge(ls9_img_col)

    return out_col


def get_sen2_s2cloudless_sr_harm_collection(
    aoi: ee.Geometry,
    start_date: datetime.datetime,
    end_date: datetime.datetime,
    cloud_thres: int = 50,
    cld_prb_thres: float = 50,
    nir_drk_thres: float = 0.15,
    cld_prj_dist: float = 1,
    clds_buffer: float = 50,
) -> ee.ImageCollection:
    sen2_start = datetime.datetime(year=2015, month=7, day=1)
    sen2_end = datetime.datetime.now()

    if not pb_gee_tools.utils.do_dates_overlap(
        s1_date=start_date, e1_date=end_date, s2_date=sen2_start, e2_date=sen2_end
    ):
        raise Exception(
            "Date range specified does not overlap "
            "with the availability of Sentinel-2 imagery."
        )

    ee_start_date = ee.Date.fromYMD(
        ee.Number(start_date.year),
        ee.Number(start_date.month),
        ee.Number(start_date.day),
    )
    ee_end_date = ee.Date.fromYMD(
        ee.Number(end_date.year), ee.Number(end_date.month), ee.Number(end_date.day)
    )

    def _add_cloud_bands(img):
        # Get s2cloudless image, subset the probability band.
        cld_prb = ee.Image(img.get("s2cloudless")).select("probability")

        # Condition s2cloudless by the probability threshold value.
        is_cloud = cld_prb.gt(cld_prb_thres).rename("clouds")

        # Add the cloud probability layer and cloud mask as image bands.
        return img.addBands(ee.Image([cld_prb, is_cloud]))

    def _add_shadow_bands(img):
        # Identify water pixels from the SCL band.
        not_water = img.select("SCL").neq(6)

        # Identify dark NIR pixels that are not water (potential cloud shadow pixels).
        SR_BAND_SCALE = 1e4
        dark_pixels = (
            img.select("B8")
            .lt(nir_drk_thres * SR_BAND_SCALE)
            .multiply(not_water)
            .rename("dark_pixels")
        )

        # Determine the direction to project cloud shadow from clouds (assumes UTM projection).
        shadow_azimuth = ee.Number(90).subtract(
            ee.Number(img.get("MEAN_SOLAR_AZIMUTH_ANGLE"))
        )

        # Project shadows from clouds for the distance specified by the CLD_PRJ_DIST input.
        cld_proj = (
            img.select("clouds")
            .directionalDistanceTransform(shadow_azimuth, cld_prj_dist * 10)
            .reproject(**{"crs": img.select(0).projection(), "scale": 100})
            .select("distance")
            .mask()
            .rename("cloud_transform")
        )

        # Identify the intersection of dark pixels with cloud shadow projection.
        shadows = cld_proj.multiply(dark_pixels).rename("shadows")

        # Add dark pixels, cloud projection, and identified shadows as image bands.
        return img.addBands(ee.Image([dark_pixels, cld_proj, shadows]))

    def _add_cld_shdw_mask(img):
        # Add cloud component bands.
        img_cloud = _add_cloud_bands(img)

        # Add cloud shadow component bands.
        img_cloud_shadow = _add_shadow_bands(img_cloud)

        # Combine cloud and shadow mask, set cloud and shadow as value 1, else 0.
        is_cld_shdw = (
            img_cloud_shadow.select("clouds")
            .add(img_cloud_shadow.select("shadows"))
            .gt(0)
        )

        # Remove small cloud-shadow patches and dilate remaining pixels by clds_buffer input.
        # 20 m scale is for speed, and assumes clouds don't require 10 m precision.
        is_cld_shdw = (
            is_cld_shdw.focalMin(2)
            .focalMax(clds_buffer * 2 / 20)
            .reproject(**{"crs": img.select([0]).projection(), "scale": 20})
            .rename("cloudmask")
        )

        # Add the final cloud-shadow mask to the image.
        return img_cloud_shadow.addBands(is_cld_shdw)

    def _apply_cld_shdw_mask(img):
        # Subset the cloudmask band and invert it so clouds/shadow are 0, else 1.
        not_cld_shdw = img.select("cloudmask").Not()

        # Subset reflectance bands and update their masks, return the result.
        return img.select("B.*").updateMask(not_cld_shdw)

    # Import and filter S2 SR.
    s2_sr_col = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(aoi)
        .filterDate(ee_start_date, ee_end_date)
        .filter(ee.Filter.lte("CLOUDY_PIXEL_PERCENTAGE", cloud_thres))
    )

    # Import and filter s2cloudless.
    s2_cloudless_col = (
        ee.ImageCollection("COPERNICUS/S2_CLOUD_PROBABILITY")
        .filterBounds(aoi)
        .filterDate(ee_start_date, ee_end_date)
    )

    # Join the filtered s2cloudless collection to the SR collection by the 'system:index' property.
    s2_sr_cld_col = ee.ImageCollection(
        ee.Join.saveFirst("s2cloudless").apply(
            **{
                "primary": s2_sr_col,
                "secondary": s2_cloudless_col,
                "condition": ee.Filter.equals(
                    **{"leftField": "system:index", "rightField": "system:index"}
                ),
            }
        )
    )

    return (
        s2_sr_cld_col.map(_add_cld_shdw_mask)
        .map(_apply_cld_shdw_mask)
        .select(["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"])
    )


def get_sen2_cloud_plus_sr_harm_collection(
    aoi: ee.Geometry,
    start_date: datetime.datetime,
    end_date: datetime.datetime,
    cloud_thres: int = 50,
    cloud_clear_thres: float = 0.60,
) -> ee.ImageCollection:
    sen2_start = datetime.datetime(year=2015, month=7, day=1)
    sen2_end = datetime.datetime.now()

    if not pb_gee_tools.utils.do_dates_overlap(
        s1_date=start_date, e1_date=end_date, s2_date=sen2_start, e2_date=sen2_end
    ):
        raise Exception(
            "Date range specified does not overlap "
            "with the availability of Sentinel-2 imagery."
        )

    ee_start_date = ee.Date.fromYMD(
        ee.Number(start_date.year),
        ee.Number(start_date.month),
        ee.Number(start_date.day),
    )
    ee_end_date = ee.Date.fromYMD(
        ee.Number(end_date.year), ee.Number(end_date.month), ee.Number(end_date.day)
    )

    s2_img_col = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
    cs_plus_img_col = ee.ImageCollection("GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED")
    qa_band = "cs_cdf"

    def _apply_s2_cloud_pls_msk(img):
        return img.updateMask(img.select(qa_band).gte(cloud_clear_thres))

    # Import and filter S2 SR.
    s2_sr_col = (
        s2_img_col.filterBounds(aoi)
        .filterDate(ee_start_date, ee_end_date)
        .filter(ee.Filter.lte("CLOUDY_PIXEL_PERCENTAGE", cloud_thres))
        .linkCollection(cs_plus_img_col, [qa_band])
        .map(_apply_s2_cloud_pls_msk)
    )

    return s2_sr_col.select(
        ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"]
    )


def get_sen2_combined_cldmsk_sr_harm_collection(
    aoi: ee.Geometry,
    start_date: datetime.datetime,
    end_date: datetime.datetime,
    cloud_thres: int = 50,
    cld_prb_thres: float = 50,
    nir_drk_thres: float = 0.15,
    cld_prj_dist: float = 1,
    clds_buffer: float = 50,
    cloud_clear_thres: float = 0.60,
) -> ee.ImageCollection:
    """
    Applies both s2cloudless and cloud+ masking

    :param aoi:
    :param start_date:
    :param end_date:
    :param cloud_thres:
    :param cld_prb_thres:
    :param nir_drk_thres:
    :param cld_prj_dist:
    :param clds_buffer:
    :param cloud_clear_thres:
    :return:
    """
    sen2_start = datetime.datetime(year=2015, month=7, day=1)
    sen2_end = datetime.datetime.now()

    if not pb_gee_tools.utils.do_dates_overlap(
        s1_date=start_date, e1_date=end_date, s2_date=sen2_start, e2_date=sen2_end
    ):
        raise Exception(
            "Date range specified does not overlap "
            "with the availability of Sentinel-2 imagery."
        )

    ee_start_date = ee.Date.fromYMD(
        ee.Number(start_date.year),
        ee.Number(start_date.month),
        ee.Number(start_date.day),
    )
    ee_end_date = ee.Date.fromYMD(
        ee.Number(end_date.year), ee.Number(end_date.month), ee.Number(end_date.day)
    )

    def _add_cloud_bands(img):
        # Get s2cloudless image, subset the probability band.
        cld_prb = ee.Image(img.get("s2cloudless")).select("probability")

        # Condition s2cloudless by the probability threshold value.
        is_cloud = cld_prb.gt(cld_prb_thres).rename("clouds")

        # Add the cloud probability layer and cloud mask as image bands.
        return img.addBands(ee.Image([cld_prb, is_cloud]))

    def _add_shadow_bands(img):
        # Identify water pixels from the SCL band.
        not_water = img.select("SCL").neq(6)

        # Identify dark NIR pixels that are not water (potential cloud shadow pixels).
        SR_BAND_SCALE = 1e4
        dark_pixels = (
            img.select("B8")
            .lt(nir_drk_thres * SR_BAND_SCALE)
            .multiply(not_water)
            .rename("dark_pixels")
        )

        # Determine the direction to project cloud shadow from clouds (assumes UTM projection).
        shadow_azimuth = ee.Number(90).subtract(
            ee.Number(img.get("MEAN_SOLAR_AZIMUTH_ANGLE"))
        )

        # Project shadows from clouds for the distance specified by the CLD_PRJ_DIST input.
        cld_proj = (
            img.select("clouds")
            .directionalDistanceTransform(shadow_azimuth, cld_prj_dist * 10)
            .reproject(**{"crs": img.select(0).projection(), "scale": 100})
            .select("distance")
            .mask()
            .rename("cloud_transform")
        )

        # Identify the intersection of dark pixels with cloud shadow projection.
        shadows = cld_proj.multiply(dark_pixels).rename("shadows")

        # Add dark pixels, cloud projection, and identified shadows as image bands.
        return img.addBands(ee.Image([dark_pixels, cld_proj, shadows]))

    def _add_cld_shdw_mask(img):
        # Add cloud component bands.
        img_cloud = _add_cloud_bands(img)

        # Add cloud shadow component bands.
        img_cloud_shadow = _add_shadow_bands(img_cloud)

        # Combine cloud and shadow mask, set cloud and shadow as value 1, else 0.
        is_cld_shdw = (
            img_cloud_shadow.select("clouds")
            .add(img_cloud_shadow.select("shadows"))
            .gt(0)
        )

        # Remove small cloud-shadow patches and dilate remaining pixels by clds_buffer input.
        # 20 m scale is for speed, and assumes clouds don't require 10 m precision.
        is_cld_shdw = (
            is_cld_shdw.focalMin(2)
            .focalMax(clds_buffer * 2 / 20)
            .reproject(**{"crs": img.select([0]).projection(), "scale": 20})
            .rename("cloudmask")
        )

        # Add the final cloud-shadow mask to the image.
        return img_cloud_shadow.addBands(is_cld_shdw)

    def _apply_cld_shdw_mask(img):
        # Subset the cloudmask band and invert it so clouds/shadow are 0, else 1.
        not_cld_shdw = img.select("cloudmask").Not()

        # Subset reflectance bands and update their masks, return the result.
        return img.select("B.*").updateMask(not_cld_shdw)

    cs_plus_img_col = ee.ImageCollection("GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED")
    qa_band = "cs_cdf"

    def _apply_s2_cloud_pls_msk(img):
        return img.updateMask(img.select(qa_band).gte(cloud_clear_thres))

    # Import and filter S2 SR.
    s2_sr_col = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(aoi)
        .filterDate(ee_start_date, ee_end_date)
        .filter(ee.Filter.lte("CLOUDY_PIXEL_PERCENTAGE", cloud_thres))
        .linkCollection(cs_plus_img_col, [qa_band])
        .map(_apply_s2_cloud_pls_msk)
    )

    # Import and filter s2cloudless.
    s2_cloudless_col = (
        ee.ImageCollection("COPERNICUS/S2_CLOUD_PROBABILITY")
        .filterBounds(aoi)
        .filterDate(ee_start_date, ee_end_date)
    )

    # Join the filtered s2cloudless collection to the SR collection by the 'system:index' property.
    s2_sr_cld_col = ee.ImageCollection(
        ee.Join.saveFirst("s2cloudless").apply(
            **{
                "primary": s2_sr_col,
                "secondary": s2_cloudless_col,
                "condition": ee.Filter.equals(
                    **{"leftField": "system:index", "rightField": "system:index"}
                ),
            }
        )
    )

    return (
        s2_sr_cld_col.map(_add_cld_shdw_mask)
        .map(_apply_cld_shdw_mask)
        .select(["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"])
    )


def get_sen1_collection(
    aoi: ee.Geometry,
    start_date: datetime.datetime,
    end_date: datetime.datetime,
    orbit_pass: int = pb_gee_tools.PB_GEE_SEN1_ASCENDING,
    add_ndpi: bool = False,
):
    def _msk_s1_edge(image):
        edge = image.lt(-40.0)
        masked_image = image.mask().And(edge.Not())
        return image.updateMask(masked_image)

    def _add_ndpi(s1_img):
        band_index_exp = "(b(0) - b(1)) / (b(0) + b(1))"
        npdi_img = s1_img.expression(band_index_exp).rename("NDPI")
        return s1_img.addBands(npdi_img)

    sen1_start = datetime.datetime(year=2014, month=4, day=1)
    sen1_end = datetime.datetime.now()

    if not pb_gee_tools.utils.do_dates_overlap(
        s1_date=start_date, e1_date=end_date, s2_date=sen1_start, e2_date=sen1_end
    ):
        raise Exception(
            "Date range specified does not overlap "
            "with the availability of Sentinel-1 imagery."
        )

    ee_start_date = ee.Date.fromYMD(
        ee.Number(start_date.year),
        ee.Number(start_date.month),
        ee.Number(start_date.day),
    )
    ee_end_date = ee.Date.fromYMD(
        ee.Number(end_date.year), ee.Number(end_date.month), ee.Number(end_date.day)
    )

    if orbit_pass == pb_gee_tools.PB_GEE_SEN1_ASCENDING:
        orbit_pass_str = "ASCENDING"
    elif orbit_pass == pb_gee_tools.PB_GEE_SEN1_DESCENDING:
        orbit_pass_str = "DESCENDING"
    else:
        raise Exception("Orbit must be ASCENDING or DESCENDING")

    s1_img_col = (
        ee.ImageCollection("COPERNICUS/S1_GRD")
        .filterBounds(aoi)
        .filterDate(ee_start_date, ee_end_date)
        .filter(ee.Filter.eq("instrumentMode", "IW"))
        .filter(ee.Filter.eq("orbitProperties_pass", orbit_pass_str))
        .select(["VV", "VH"])
        .map(_msk_s1_edge)
    )

    if add_ndpi:
        s1_img_col = s1_img_col.map(_add_ndpi)

    return s1_img_col


def get_modis_albedo_collection(
    aoi: ee.Geometry,
    start_date: datetime.datetime,
    end_date: datetime.datetime,
):
    def _add_vld_bands(img):
        # Get quality bands
        b1_quality = ee.Image(img.get("modis_MCD43")).select(
            "BRDF_Albedo_Band_Quality_Band1"
        )
        b2_quality = ee.Image(img.get("modis_MCD43")).select(
            "BRDF_Albedo_Band_Quality_Band2"
        )
        b3_quality = ee.Image(img.get("modis_MCD43")).select(
            "BRDF_Albedo_Band_Quality_Band3"
        )
        b4_quality = ee.Image(img.get("modis_MCD43")).select(
            "BRDF_Albedo_Band_Quality_Band4"
        )
        b5_quality = ee.Image(img.get("modis_MCD43")).select(
            "BRDF_Albedo_Band_Quality_Band5"
        )
        b6_quality = ee.Image(img.get("modis_MCD43")).select(
            "BRDF_Albedo_Band_Quality_Band6"
        )
        b7_quality = ee.Image(img.get("modis_MCD43")).select(
            "BRDF_Albedo_Band_Quality_Band7"
        )

        # Add quality bands to the image.
        return img.addBands(
            ee.Image(
                [
                    b1_quality,
                    b2_quality,
                    b3_quality,
                    b4_quality,
                    b5_quality,
                    b6_quality,
                    b7_quality,
                ]
            )
        )

    def _mask_qa_combined(img):
        qa_b1_mask = img.select("BRDF_Albedo_Band_Quality_Band1").lt(3)
        qa_b2_mask = img.select("BRDF_Albedo_Band_Quality_Band2").lt(3)
        qa_b3_mask = img.select("BRDF_Albedo_Band_Quality_Band3").lt(3)
        qa_b4_mask = img.select("BRDF_Albedo_Band_Quality_Band4").lt(3)
        qa_b5_mask = img.select("BRDF_Albedo_Band_Quality_Band5").lt(3)
        qa_b6_mask = img.select("BRDF_Albedo_Band_Quality_Band6").lt(3)
        qa_b7_mask = img.select("BRDF_Albedo_Band_Quality_Band7").lt(3)

        return (
            img.updateMask(qa_b1_mask)
            .updateMask(qa_b2_mask)
            .updateMask(qa_b3_mask)
            .updateMask(qa_b4_mask)
            .updateMask(qa_b5_mask)
            .updateMask(qa_b6_mask)
            .updateMask(qa_b7_mask)
        )

    modis_alb_img_col = (
        ee.ImageCollection("MODIS/061/MCD43A3")
        .filterBounds(aoi)
        .filterDate(start_date, end_date)
    )
    modis_alb_qual_img_col = (
        ee.ImageCollection("MODIS/061/MCD43A2")
        .filterBounds(aoi)
        .filterDate(start_date, end_date)
    )

    modis_mcd43_img_col = ee.ImageCollection(
        ee.Join.saveFirst("modis_MCD43").apply(
            **{
                "primary": modis_alb_img_col,
                "secondary": modis_alb_qual_img_col,
                "condition": ee.Filter.equals(
                    **{
                        "leftField": "system:time_start",
                        "rightField": "system:time_start",
                    }
                ),
            }
        )
    )

    modis_img_qual_col = modis_mcd43_img_col.map(_add_vld_bands)
    modis_img_qual_mskd_col = modis_img_qual_col.map(_mask_qa_combined)

    return modis_img_qual_mskd_col.select(
        [
            "Albedo_BSA_Band1",
            "Albedo_BSA_Band2",
            "Albedo_BSA_Band3",
            "Albedo_BSA_Band4",
            "Albedo_BSA_Band5",
            "Albedo_BSA_Band6",
            "Albedo_BSA_Band7",
            "Albedo_BSA_vis",
            "Albedo_BSA_nir",
            "Albedo_BSA_shortwave",
            "Albedo_WSA_Band1",
            "Albedo_WSA_Band2",
            "Albedo_WSA_Band3",
            "Albedo_WSA_Band4",
            "Albedo_WSA_Band5",
            "Albedo_WSA_Band6",
            "Albedo_WSA_Band7",
            "Albedo_WSA_vis",
            "Albedo_WSA_nir",
            "Albedo_WSA_shortwave",
        ]
    )
