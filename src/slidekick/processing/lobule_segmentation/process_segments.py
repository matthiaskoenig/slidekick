import pickle
from pathlib import Path
from typing import List, Tuple, Union

import shapely
from shapely import LineString, GeometryCollection, Polygon, make_valid, polygonize_full, affinity, Geometry

from slidekick.processing.lobule_segmentation.get_segments import LineSegmentsFinder


def get_polys(geometry: Union[GeometryCollection | Geometry], polygon_list: list):
    if isinstance(geometry, GeometryCollection):
        for geom in geometry.geoms:
            get_polys(geom, polygon_list)
    else:
        if isinstance(geometry, Polygon):
            polygon_list.append(geometry)


def translate_polygon(poly: shapely.Polygon, pad: int) -> Polygon:
    return affinity.translate(poly, xoff=-pad, yoff=-pad)


def process_line_segments(line_segments: List[List[Tuple[int, int]]],
                          vessel_classes: List[int],
                          vessel_contours: list,
                          pad: int) -> SlideStats:
    linestrings = [LineString(s) for s in line_segments]

    vessel_polys = [Polygon(cont.squeeze(axis=1).tolist()) for cont in vessel_contours if len(cont) >= 4]
    vessel_polys = [Polygon([(y, x) for x, y in poly.exterior.coords]) for poly in vessel_polys]

    vessel_polys = [p if p.is_valid else make_valid(p) for p in vessel_polys]

    # result = shapely.multipolygons(shapely.get_parts(polygonize(linestrings)))
    valid, cut_edges, dangles, invalid_rings = polygonize_full(linestrings)
    # print(len(valid.geoms), len(cut_edges.geoms), len(dangles.geoms), len(invalid_rings.geoms))

    # create polygon from the fucked up line strings and make valid
    made_valid = GeometryCollection([make_valid(Polygon(geom)) for geom in invalid_rings.geoms])
    # filter the remaining valid polygons

    made_valid_polys = []
    get_polys(made_valid, made_valid_polys)

    made_valid = GeometryCollection(made_valid_polys)

    result = shapely.multipolygons(shapely.get_parts(valid))

    vessels = []
    lobuli = []

    for geo_collection in [result, made_valid]:
        for poly in geo_collection.geoms:
            for vessel_poly in vessel_polys:
                if vessel_poly.buffer(2.0).contains(poly):
                    vessels.append(poly)
                    break
            else:
                lobuli.append(poly)

    class_0: List[Polygon] = [p for p, c in zip(vessel_polys, vessel_classes) if c == 0]
    class_1: List[Polygon] = [p for p, c in zip(vessel_polys, vessel_classes) if c == 1]
    unclassified: List[Polygon] = [p for p, c in zip(vessel_polys, vessel_classes) if c is None]

    stats = []

    # translate polygons back by the pad applied in the clustering algorithm.
    lobuli = [translate_polygon(p, pad) for p in lobuli]
    class_1 = [translate_polygon(p, pad) for p in class_1]
    class_0 = [translate_polygon(p, pad) for p in class_0]
    unclassified = [translate_polygon(p, pad) for p in unclassified]

    for k, lobulus_poly in enumerate(lobuli):
        c0, c1, uc = [], [], []
        c0_idx, c1_idx, uc_idx = [], [], []
        for i, p in enumerate(class_0):
            if lobulus_poly.contains(p) or lobulus_poly.intersects(p):
                c0.append(p)
                c0_idx.append(i)

        for i, p in enumerate(class_1):
            if lobulus_poly.contains(p) or lobulus_poly.intersects(p):
                c1.append(p)
                c1_idx.append(i)

        for i, p in enumerate(unclassified):
            if lobulus_poly.contains(p) or lobulus_poly.intersects(p):
                uc.append(p)
                uc_idx.append(i)

        stats.append(LobuleStatistics.from_polygon(k, lobulus_poly, c0, c1, uc, c0_idx, c1_idx, uc_idx))

    return SlideStats(stats, class_0, class_1, unclassified)

