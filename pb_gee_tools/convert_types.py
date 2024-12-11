import ee


def get_vec_file_bbox_wgs84(vec_file, vec_lyr=None):
    """
    A function which creates a bounding box (ee.geometry.BBox) ee object
    representing the bounding box of the input vector file.

    :param vec_file: file path for the vector file
    :param vec_lyr: vector layer name
    :return: ee.Geometry.BBox

    """
    import geopandas

    # Read the vector layer and make sure it is project using WGS84 (EPSG:4326)
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
    using an region of interest polygon.

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
    import geopandas

    # Read the vector layer and make sure it is project using WGS84 (EPSG:4326)
    vec_gdf = geopandas.read_file(vec_file, layer=vec_lyr).to_crs(4326)

    if vec_gdf.geom_type[0] != "Point":
        raise Exception("Input layer needs to be Point")

    if vec_roi_file is not None:
        vec_roi_gdf = geopandas.read_file(vec_roi_file, layer=vec_roi_lyr).to_crs(4326)
        if vec_roi_gdf.geom_type[0] != "Polygon":
            raise Exception("Input ROI layer needs to be Polygon")

        vec_gdf = vec_gdf.clip(vec_roi_gdf)

    if rnd_smpl is not None:
        vec_gdf = vec_gdf.sample(n=rnd_smpl, replace=use_replace, random_state=rnd_seed)

    # Get all point coordinates from input file.
    coords = vec_gdf.get_coordinates(
        include_z=False, ignore_index=True, index_parts=False
    )

    pts = list()
    for row in coords.iterrows():
        # print(row[1].x, row[1].y)
        pts.append(ee.Geometry.Point([row[1].x, row[1].y]))
    gee_pts = ee.Geometry.MultiPoint(pts)
    return gee_pts
