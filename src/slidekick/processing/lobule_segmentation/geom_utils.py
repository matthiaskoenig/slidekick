from typing import List, Tuple, Union

import cv2
import matplotlib.pyplot as plt
import numpy as np
from shapely.ops import transform
from shapely import Geometry, Polygon, LineString, LinearRing, Point, GeometryCollection
from shapely.geometry.base import BaseGeometry, BaseMultipartGeometry

from slidekick.console import console

# copied from /zia/pipeline/common/geometry_utils.py

def off_set_geometry(geometry: Geometry, offset: Tuple[int, int]):
    return transform(lambda x, y: (x - offset[0], y - offset[1]), geometry)

class AxGeometryDraw:

    @classmethod
    def draw_geometry(
            cls,
            ax: plt.Axes,
            geometry: BaseMultipartGeometry,
            offset: Tuple[int, int] = (0, 0),
            **kwargs,
    ) -> None:
        if isinstance(geometry, BaseMultipartGeometry):
            cls._draw_multipart_recursively(geometry, ax, offset, **kwargs)

        elif isinstance(geometry, BaseGeometry):
            cls._draw_geometry(geometry, ax, offset, **kwargs)
        else:
            console.print(f"Non polygon type geometry encountered {type(geometry)}", style="warning")

    @classmethod
    def _draw_polygon(
            cls, polygon: Polygon, ax: plt.Axes, **kwargs
    ) -> None:
        if polygon.is_empty:
            return

        fc = kwargs.get("facecolor")
        ec = kwargs.get("edgecolor")
        lw = kwargs.get("linewidth")

        x, y = polygon.exterior.xy
        ax.fill(y, x, facecolor=fc, edgecolor=ec, linewidth=lw)

    @classmethod
    def _draw_line_string(
            cls,
            line_string: LineString,
            ax: plt.Axes,
            **kwargs,
    ) -> None:
        if line_string.is_empty:
            return

        fc = kwargs.get("facecolor")
        lw = kwargs.get("linewidth")

        x, y = line_string.xy

        ax.plot(y, x, color=fc, linewidth=lw)

    @classmethod
    def _draw_point(cls, point: Point, ax: plt.Axes, **kwargs) -> None:
        if point.is_empty:
            return

        fc = kwargs.get("facecolor")
        ec = kwargs.get("edgecolor")
        lw = kwargs.get("linewidth")
        ms = kwargs.get("markersize")

        x, y = point.xy

        ax.scatter(y, x, c=fc, edgecolors=ec, linewidths=lw, s=ms)

    @classmethod
    def _draw_multipart_recursively(cls, geometry: BaseMultipartGeometry, ax: plt.Axes, offset: Tuple[int, int], **kwargs):
        for geometry in geometry.geoms:
            if isinstance(geometry, BaseMultipartGeometry):
                cls._draw_multipart_recursively(geometry, ax, offset, **kwargs)
            else:
                cls._draw_geometry(geometry, ax, offset, **kwargs)

    @classmethod
    def _draw_geometry(cls, geometry: BaseGeometry, ax: plt.Axes, offset: Tuple[int, int], **kwargs):
        geometry = off_set_geometry(geometry, offset)
        if isinstance(geometry, Polygon):
            cls._draw_polygon(geometry, ax, **kwargs)
        elif isinstance(geometry, LineString):
            cls._draw_line_string(geometry, ax, **kwargs)
        elif isinstance(geometry, Point):
            cls._draw_point(geometry, ax, **kwargs)
        else:
            console.print(f"Encountered geometry type {type(geometry)} is not supported.", style = "warning")