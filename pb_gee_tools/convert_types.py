import os
import ee
from typing import Union, List, Tuple
import geopandas


def get_vec_file_bbox_wgs84(vec_file, vec_lyr=None):
    """
    :param vec_file: A file path to the vector file that contains the spatial data.
    :param vec_lyr: The name of the layer within the vector file to be used
                    for processing. Default is None.
    :return: An Earth Engine Geometry object representing the bounding box of the
             vector data in WGS84 (EPSG:4326) projection.

    """
    # Read the vector layer and make sure it is project using WGS84 (EPSG:4326)
    if "parquet" in os.path.basename(vec_file):
        vec_gdf = geopandas.read_parquet(vec_file)
    else:
        vec_gdf = geopandas.read_file(vec_file, layer=vec_lyr).to_crs(4326)

    # Get layer bbox: minx, miny, maxx, maxy
    gp_bbox = vec_gdf.total_bounds

    # Create the GEE geometry from the bbox.
    roi_west = gp_bbox[0]
    roi_east = gp_bbox[2]
    roi_north = gp_bbox[3]
    roi_south = gp_bbox[1]
    vec_bbox_aoi = ee.Geometry.BBox(roi_west, roi_south, roi_east, roi_north)
    return vec_bbox_aoi


def convert_vector_to_gee_polygon(vec_file: str, vec_lyr: str = None):
    """
    Converts a vector file with a single polygon to a ee.Geometry.Polygon object.
    :param vec_file: file path to the vector file
    :param vec_lyr: vector layer name
    :return: an ee.Geometry.Polygon object
    """

    # Check the vector file extension
    if "parquet" in os.path.basename(vec_file):
        vec_gdf = geopandas.read_parquet(vec_file)
    else:
        vec_gdf = geopandas.read_file(vec_file, layer=vec_lyr).to_crs(4326)

    if vec_gdf.geom_type.iat[0] != "Polygon":
        raise Exception("Input layer needs to be Polygon")

    # Convert the polygon to an Earth Engine object
    polygon = ee.Geometry.Polygon(list(map(list, vec_gdf.geometry[0].exterior.coords)))

    return polygon


def get_gee_multi_polygon(vec_file: str, vec_lyr: str = None):
    """
    A function which converts an input vector file into
    a ee.Geometry.MultiPolygon object.

    :param vec_file: file path to the vector file
    :param vec_lyr: vector layer name
    :return: ee.Geometry.MultiPolygon
    """

    # Check the vector file extension
    if "parquet" in os.path.basename(vec_file):
        vec_gdf = geopandas.read_parquet(vec_file)
    else:
        vec_gdf = geopandas.read_file(vec_file, layer=vec_lyr).to_crs(4326)

    if vec_gdf.geom_type.iat[0] != "Polygon":
        raise Exception("Input layer needs to be Polygon")

    # Convert polygons to Earth Engine objects
    polygons = [
        ee.Geometry.Polygon(list(map(list, p.exterior.coords)))
        for p in vec_gdf.geometry
    ]

    # Return a ee.Geometry.MultiPolygon
    return ee.Geometry.MultiPolygon(polygons)


def get_gee_pts(
    vec_file: str,
    vec_lyr: str = None,
    rnd_smpl: int = None,
    rnd_seed: int = None,
    use_replace: bool = False,
    vec_roi_file: str = None,
    vec_roi_lyr: str = None,
):
    """
    A function which converts an input vector file into a ee.geometry.MultiPoint
    object. This function is intended to be used to when reading a set of points
    as training data for classification. If provided the points will be subsetted
    using a region of interest polygon.

    :param vec_file: file path for the vector file which must be of type Points.
    :param vec_lyr: vector layer name
    :param rnd_smpl: Optionally randomly sample the data to retreve a subset of the
                     input options. (e.g., return 100 points)
    :param rnd_seed: Optionally seed the random number generator for reproducibility.
    :param use_replace: Optionally use replacement when randomly sampling the
                        vector layer.
    :param vec_roi_file: file path for the vector file defining a region of interest.
    :param vec_roi_lyr: vector layer name of the region of interest.
    :return: ee.Geometry.MultiPoint

    """
    if vec_roi_file is not None:
        if "parquet" in os.path.basename(vec_roi_file):
            vec_roi_gdf = geopandas.read_parquet(vec_roi_file)
        else:
            vec_roi_gdf = geopandas.read_file(vec_roi_file, layer=vec_roi_lyr).to_crs(
                4326
            )
        if vec_roi_gdf.geom_type.iat[0] != "Polygon":
            raise Exception("Input ROI layer needs to be Polygon")
    else:
        vec_roi_gdf = None

    return get_gee_pts_gp(
        vec_file, vec_lyr, rnd_smpl, rnd_seed, use_replace, vec_roi_gdf
    )


def get_gee_pts_bbox(
    vec_file: str,
    vec_lyr: str = None,
    rnd_smpl: int = None,
    rnd_seed: int = None,
    use_replace: bool = False,
    bbox: Union[Tuple[float, float, float, float], List[float]] = None,
):
    """
    A function which converts an input vector file into a ee.geometry.MultiPoint
    object. This function is intended to be used to when reading a set of points
    as training data for classification. If provided the points will be subsetted
    using a region of interest polygon.

    :param vec_file: file path for the vector file which must be of type Points.
    :param vec_lyr: vector layer name
    :param rnd_smpl: Optionally randomly sample the data to retreve a subset of the
                     input options. (e.g., return 100 points)
    :param rnd_seed: Optionally seed the random number generator for reproducibility.
    :param use_replace: Optionally use replacement when randomly sampling the
                        vector layer.
    :param bbox: the bounding box (xMin, xMax, yMin, yMax): EPSG:4326.
    :return: ee.Geometry.MultiPoint

    """
    from shapely.geometry import Polygon

    if bbox is not None:
        polygons = [
            Polygon(
                [
                    (bbox[0], bbox[2]),
                    (bbox[0], bbox[3]),
                    (bbox[1], bbox[3]),
                    (bbox[1], bbox[2]),
                ]
            )
        ]
        vec_roi_gdf = geopandas.GeoDataFrame({"geometry": polygons})
        vec_roi_gdf = vec_roi_gdf.set_crs("EPSG:4326", allow_override=True)
    else:
        vec_roi_gdf = None

    return get_gee_pts_gp(
        vec_file, vec_lyr, rnd_smpl, rnd_seed, use_replace, vec_roi_gdf
    )


def get_gee_pts_gp(
    vec_file: str,
    vec_lyr: str = None,
    rnd_smpl: int = None,
    rnd_seed: int = None,
    use_replace: bool = False,
    gp_roi_gdf: geopandas.GeoDataFrame = None,
):
    """
    A function which converts an input vector file into a ee.geometry.MultiPoint
    object. This function is intended to be used to when reading a set of points
    as training data for classification. If provided the points will be subsetted
    using a region of interest polygon.

    :param vec_file: file path for the vector file which must be of type Points.
    :param vec_lyr: vector layer name
    :param rnd_smpl: Optionally randomly sample the data to retreve a subset of the
                     input options. (e.g., return 100 points)
    :param rnd_seed: Optionally seed the random number generator for reproducibility.
    :param use_replace: Optionally use replacement when randomly sampling the
                        vector layer.
    :param gp_roi_gdf: geopandas object
    :return: ee.Geometry.MultiPoint

    """
    # Read the vector layer and make sure it is project using WGS84 (EPSG:4326)
    if "parquet" in os.path.basename(vec_file):
        vec_gdf = geopandas.read_parquet(vec_file).to_crs(4326)
    else:
        vec_gdf = geopandas.read_file(vec_file, layer=vec_lyr).to_crs(4326)

    return get_gee_pts_gp_gdf(
        vec_gdf,
        rnd_smpl,
        rnd_seed,
        use_replace,
        gp_roi_gdf,
    )


def get_gee_pts_gp_gdf(
    vec_gdf: geopandas.GeoDataFrame,
    rnd_smpl: int = None,
    rnd_seed: int = None,
    use_replace: bool = False,
    gp_roi_gdf: geopandas.GeoDataFrame = None,
):
    """
    A function which converts an input vector file into a ee.geometry.MultiPoint
    object. This function is intended to be used to when reading a set of points
    as training data for classification. If provided the points will be subsetted
    using a region of interest polygon.

    :param vec_gdf: geopandas GeoDataframe of the type Points.
    :param rnd_smpl: Optionally randomly sample the data to retreve a subset of the
                     input options. (e.g., return 100 points)
    :param rnd_seed: Optionally seed the random number generator for reproducibility.
    :param use_replace: Optionally use replacement when randomly sampling the
                        vector layer.
    :param gp_roi_gdf: geopandas object
    :return: ee.Geometry.MultiPoint

    """
    if vec_gdf.geom_type.iat[0] != "Point":
        raise Exception("Input layer needs to be Point")

    if gp_roi_gdf is not None:
        if gp_roi_gdf.geom_type.iat[0] != "Polygon":
            raise Exception("Input ROI layer needs to be Polygon")

        vec_gdf = vec_gdf.clip(gp_roi_gdf)

    if rnd_smpl is not None:
        vec_gdf = vec_gdf.sample(n=rnd_smpl, replace=use_replace, random_state=rnd_seed)

    # Get all point coordinates from input file.
    coords = vec_gdf.get_coordinates(
        include_z=False, ignore_index=True, index_parts=False
    )

    pts = list()
    for row in coords.iterrows():
        pts.append(ee.Geometry.Point([row[1].x, row[1].y]))
    gee_pts = ee.Geometry.MultiPoint(pts)
    return gee_pts
