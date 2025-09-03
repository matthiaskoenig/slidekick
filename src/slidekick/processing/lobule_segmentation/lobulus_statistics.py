"""
Takes list of polygons and calculates statistics on them.

- area
- number of corners (after useful simplification)
- circumference length
- symmetry

"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Union, Optional, Tuple, Dict

import geojson
import numpy as np
import pandas as pd
import shapely
import shapely.ops
from matplotlib import pyplot as plt
from shapely import Polygon, MultiPolygon, GeometryCollection, minimum_bounding_radius, Geometry
from shapely.geometry import shape

from slidekick.processing.lobule_segmentation.geom_utils import AxGeometryDraw

# Mostly copy of zia/pipeline/pipeline_components/algorithm/segementation/lobulus_statistics.py


def offset_geom(geometry: Geometry, offset: Tuple[int, int]):
    return shapely.ops.transform(lambda x, y: (x - offset[0], y - offset[1]), geometry)


@dataclass(init=True)
class SlideStats:
    lobule_stats: List[LobuleStatistics]
    vessels_central: List[Union[Polygon, MultiPolygon, GeometryCollection]]
    vessels_portal: List[Union[Polygon, MultiPolygon, GeometryCollection]]
    unclassified: List[Union[Polygon, MultiPolygon, GeometryCollection]]

    def _vessels_central_to_geojson(self, result_dir: Path) -> None:
        features = [geojson.Feature(geometry=g.__geo_interface__) for g in self.vessels_central]
        col = geojson.FeatureCollection(features=features)
        with open(result_dir / "central_vessels.geojson", "w") as f:
            geojson.dump(col, f)

    def _vessels_portal_to_geojson(self, result_dir: Path) -> None:
        features = [geojson.Feature(geometry=g.__geo_interface__) for g in self.vessels_portal]
        col = geojson.FeatureCollection(features=features)
        with open(result_dir / "portal_vessels.geojson", "w") as f:
            geojson.dump(col, f)

    def _vessels_unclassified_to_geojson(self, result_dir: Path) -> None:
        features = [geojson.Feature(geometry=g.__geo_interface__) for g in self.unclassified]
        col = geojson.FeatureCollection(features=features)
        with open(result_dir / "unclassified_vessels.geojson", "w") as f:
            geojson.dump(col, f)

    def _lobuli_polygons_to_geojson(self, result_dir: Path) -> None:
        # saving the lobuli polygons as a feature collection
        lobule_features = []
        for i, lobule_stat in enumerate(self.lobule_stats):
            lf: geojson.Feature = lobule_stat.to_geo_json_feature()
            lobule_features.append(lf)

        col = geojson.FeatureCollection(features=lobule_features)
        col["metaData"] = self.meta_data
        with open(result_dir / "lobuli.geojson", "w") as f:
            geojson.dump(col, f)

    def _lobule_local_maxima_to_geojson(self, result_dir: Path) -> None:
        lobule_local_maxima = []
        for i, lobule_stat in enumerate(self.lobule_stats):
            geo_col = lobule_stat.local_max_to_geometry_collection()
            if geo_col is not None:
                lobule_local_maxima.append(geo_col)

        if len(lobule_local_maxima) > 0:
            col = geojson.FeatureCollection(features=lobule_local_maxima)
            with open(result_dir / "local_maxima.geojson", "w") as f:
                geojson.dump(col, f)

    def to_geojson(self, result_dir: Path) -> None:
        result_dir.mkdir(parents=True, exist_ok=True)

        # saving the vessels of the slide as a feature collection
        self._vessels_central_to_geojson(result_dir)
        self._vessels_portal_to_geojson(result_dir)
        self._vessels_unclassified_to_geojson(result_dir)

        # saving the lobuli polygons as a feature collection
        self._lobuli_polygons_to_geojson(result_dir)

        # save local maxima of lobuli as geometry collections in a feature collection
        self._lobule_local_maxima_to_geojson(result_dir)

    @classmethod
    def load_central_vessels(cls, result_dir: Path) -> List[Polygon]:
        central_vessel_dir = result_dir / "central_vessels.geojson"
        if not central_vessel_dir.exists():
            raise FileNotFoundError(f"{central_vessel_dir} does not exist.")

        with open(central_vessel_dir, "r") as f:
            col = geojson.load(f)

        return [shape(feature["geometry"]) for feature in col["features"]]

    @classmethod
    def load_portal_vessels(cls, result_dir: Path) -> List[Polygon]:
        portal_vessel_dir = result_dir / "portal_vessels.geojson"
        if not portal_vessel_dir.exists():
            raise FileNotFoundError(f"{portal_vessel_dir} does not exist.")

        with open(portal_vessel_dir, "r") as f:
            col = geojson.load(f)

        return [shape(feature["geometry"]) for feature in col["features"]]

    @classmethod
    def load_unclassified_vessels(cls, result_dir: Path) -> List[Polygon]:
        unclassified_dir = result_dir / "unclassified_vessels.geojson"
        if not unclassified_dir.exists():
            raise FileNotFoundError(f"{unclassified_dir} does not exist.")

        with open(unclassified_dir, "r") as f:
            col = geojson.load(f)

        return [shape(feature["geometry"]) for feature in col["features"]]

    @classmethod
    def load_local_maxima(cls, result_dir) -> Optional[Dict[int, GeometryCollection]]:
        unclassified_dir = result_dir / "local_maxima.geojson"
        if not unclassified_dir.exists():
            # logger.warning("No geojson file for local lobule maxima found.")
            return None

        with open(unclassified_dir, "r") as f:
            col = geojson.load(f)

        return {feature["properties"]["lobule_id"]: shape(feature["geometry"]) for feature in col["features"]}

    @classmethod
    def load_from_file_system(cls, result_dir: Path) -> SlideStats:
        if not result_dir.exists():
            raise FileNotFoundError("The specified location was not found.")

        central_vessels = cls.load_central_vessels(result_dir)
        portal_vessels = cls.load_portal_vessels(result_dir)
        unclassified = cls.load_unclassified_vessels(result_dir)
        local_maxima = cls.load_local_maxima(result_dir)

        lobule_dir = result_dir / "lobuli.geojson"
        if not lobule_dir.exists():
            raise FileNotFoundError(f"{lobule_dir} does not exist.")

        with open(lobule_dir, "r") as f:
            col = geojson.load(f)

        lobule_stats = []

        for feature in col["features"]:
            central_vessels_idx = feature["properties"]["central_vessels"]
            portal_vessels_idx = feature["properties"]["portal_vessels"]
            unclassified_idx = feature["properties"]["unclassified"]
            id_ = feature["properties"]["lobule_id"]

            lobule_stat = LobuleStatistics.from_polygon(
                id_,
                shape(feature["geometry"]),
                [central_vessels[i] for i in central_vessels_idx],
                [portal_vessels[i] for i in portal_vessels_idx],
                [unclassified[i] for i in unclassified_idx],
                central_vessels_idx,
                portal_vessels_idx,
                unclassified_idx,
                local_maxima[id_] if local_maxima is not None else None
            )

            lobule_stats.append(lobule_stat)

        return cls(lobule_stats, central_vessels, portal_vessels, unclassified, col.get("metaData"))

    def plot(self, report_path=None):
        fig, ax = plt.subplots(1, 1, dpi=300)
        ax: plt.Axes
        colors = np.random.rand(len(self.lobule_stats), 3)  # Random RGB values between 0 and 1
        for i, stat in enumerate(self.lobule_stats):
            x, y = stat.polygon.exterior.xy
            ax.fill(y, x, facecolor=colors[i], edgecolor="black", linewidth=0.2)

            if stat.local_maxima is not None:
                AxGeometryDraw.draw_geometry(ax, stat.local_maxima, facecolor="grey", edgecolor="grey", linewidth=0.2, markersize=0.2)

        for i, geom in enumerate(self.vessels_central):
            x, y = geom.buffer(1.0).exterior.xy
            ax.fill(y, x, facecolor="black", edgecolor="black", linewidth=0.2)

        for i, geom in enumerate(self.vessels_portal):
            x, y = geom.buffer(1.0).exterior.xy
            ax.fill(y, x, facecolor="white", edgecolor="black", linewidth=0.2)

        # ax.set_xlim(right=labels.shape[1])
        # ax.set_ylim(top=labels.shape[0])

        ax.set_title(f"{self.meta_data.get('subject')} {self.meta_data.get('lobe_id')}")

        ax.set_aspect("equal")
        ax.invert_yaxis()

        ax.axis("off")

        if report_path is not None:
            plt.savefig(report_path / "polygons.png", bbox_inches="tight")

        plt.show()

    def plot_on_axis(self, ax: plt.Axes,
                     lobulus_ec: str = "lime",
                     lobulus_fc: Optional[str] = None,
                     lobulus_alpha: float = 1,
                     pvessel_ec: str = "magenta",
                     pvessel_fc: Optional[str] = "magenta",
                     pvessel_alpha: float = 0.5,
                     cvessel_ec: str = "cyan",
                     cvessel_fc: Optional[str] = "cyan",
                     cvessel_alpha: float = 0.5,
                     ucvessel_ec: str = "black",
                     ucvessel_fc: Optional[str] = "black",
                     ucvessel_alpha: float = 0.5,
                     plot_max=False,
                     lmaxima_ec: str = "grey",
                     lmaxima_fc: str = "grey",
                     linewidth=0.5,
                     offset: Tuple = None,
                     enumerate_=False
                     ):

        for i, stat in enumerate(self.lobule_stats):
            if offset is not None:
                polygon = offset_geom(stat.polygon, offset)
            else:
                polygon = stat.polygon

            x, y = polygon.exterior.xy
            ax.fill(y, x, facecolor=lobulus_fc if lobulus_fc is not None else "none", edgecolor=lobulus_ec, alpha=lobulus_alpha, linewidth=linewidth)

            if enumerate_:
                ax.text(s=f"{i}", x=polygon.centroid.y, y=polygon.centroid.x, va="center", ha="center", fontsize=6)

            if plot_max:
                AxGeometryDraw.draw_geometry(ax, stat.local_maxima, edgecolor=lmaxima_ec, facecolor=lmaxima_fc, linewidth=linewidth,
                                             markersize=linewidth)

        for i, geom in enumerate(self.vessels_central):
            if offset:
                geom = offset_geom(geom, offset)
            x, y = geom.buffer(1.0).exterior.xy
            ax.fill(y, x, facecolor=cvessel_fc, edgecolor=cvessel_ec, alpha=cvessel_alpha, linewidth=linewidth)

        for i, geom in enumerate(self.vessels_portal):
            if offset:
                geom = offset_geom(geom, offset)
            x, y = geom.buffer(1.0).exterior.xy
            ax.fill(y, x, facecolor=pvessel_fc, edgecolor=pvessel_ec, alpha=pvessel_alpha, linewidth=linewidth)

        for i, geom in enumerate(self.unclassified):
            if offset:
                geom = offset_geom(geom, offset)
            x, y = geom.buffer(1.0).exterior.xy
            ax.fill(y, x, facecolor=ucvessel_fc, edgecolor=ucvessel_ec, alpha=ucvessel_alpha, linewidth=linewidth)

    def to_dataframe(self) -> pd.DataFrame:
        rows = []
        pixel_size = self.meta_data["pixel_size"]
        level = self.meta_data["level"]

        dimension_factor = pixel_size * 2 ** level

        for stat in self.lobule_stats:
            row_dict = dict(
                area=stat.get_area() * dimension_factor ** 2,
                area_unit="µm$^2$",
                perimeter=stat.get_perimeter(),
                perimeter_unit="µm",
                n_central_vessel=len(stat.vessels_central),
                n_portal_vessel=len(stat.vessels_portal),
                central_vessel_cross_section=stat.get_central_vessel_cross_section() * dimension_factor ** 2,
                central_vessel_cross_section_unit="µm$^$2",
                portal_vessel_cross_section=stat.get_portal_vessel_cross_section() * dimension_factor ** 2,
                portal_vessel_cross_section_unit="µm$^$2",
                compactness=stat.get_compactness(),
                compactness_unit="-",
                area_without_vessels=stat.get_poly_area_without_vessel_area() * dimension_factor ** 2,
                area_without_vessels_unit="µm$^$2",
                minimum_bounding_radius=stat.get_enclosing_circle_radius() * dimension_factor,
                minimum_bounding_radius_unit="µm"
            )
            rows.append(row_dict)
        return pd.DataFrame(rows)


@dataclass
class LobuleStatistics:
    _id: int
    polygon: Polygon
    vessels_central: List[Union[Polygon, MultiPolygon, GeometryCollection]]
    vessels_portal: List[Union[Polygon, MultiPolygon, GeometryCollection]]
    unclassified: List[Union[Polygon, MultiPolygon, GeometryCollection]]
    vessels_central_idx: List[int]
    vessels_portal_idx: List[int]
    unclassified_idx: List[int]
    local_maxima: GeometryCollection = None

    @classmethod
    def from_polygon(cls,
                     id: int,
                     polygon: Polygon,
                     vessels_central: List[Polygon],
                     vessels_portal: List[Polygon],
                     unclassified: List[Polygon],
                     vessels_central_idx: List[int],
                     vessels_portal_idx: List[int],
                     unclassified_idx: List[int],
                     local_maxima: GeometryCollection = None
                     ) -> LobuleStatistics:

        lobule_statistics = LobuleStatistics(
            _id=id,
            polygon=polygon,
            vessels_central=vessels_central,
            vessels_portal=vessels_portal,
            unclassified=unclassified,
            vessels_central_idx=vessels_central_idx,
            vessels_portal_idx=vessels_portal_idx,
            unclassified_idx=unclassified_idx,
            local_maxima=local_maxima
        )

        return lobule_statistics

    def get_area(self) -> float:
        return Polygon(self.polygon.exterior).area

    def has_central_vessel(self) -> bool:
        return len(self.vessels_central) != 0

    def has_portal_vessel(self) -> bool:
        return len(self.vessels_portal) != 0

    def get_central_vessel_cross_section(self) -> float:
        if not self.has_central_vessel():
            return 0.0
        else:
            return sum([g.area for g in self.vessels_central])

    def get_portal_vessel_cross_section(self) -> float:
        if not self.has_portal_vessel():
            return 0.0
        else:
            return sum([g.area for g in self.vessels_portal])

    def get_poly_area_without_vessel_area(self) -> float:
        return self.polygon.area

    def get_perimeter(self) -> float:
        return Polygon(self.polygon.exterior).length

    def get_compactness(self) -> float:
        """isoperimetric quotient -> ratio of the area of the shape to the area of a circle with the same perimeter"""
        area = self.polygon.area
        perimeter = self.get_perimeter()

        return area * 4 * np.pi / perimeter ** 2

    def get_enclosing_circle_radius(self) -> float:
        return minimum_bounding_radius(self.polygon)

    def to_geo_json_feature(self) -> geojson.Feature:
        return geojson.Feature(geometry=self.polygon.__geo_interface__,
                               properties={
                                   "central_vessels": self.vessels_central_idx,
                                   "portal_vessels": self.vessels_portal_idx,
                                   "unclassified": self.unclassified_idx,
                                   "lobule_id": self._id
                               })

    def local_max_to_geometry_collection(self) -> Optional[geojson.Feature]:
        if self.local_maxima is None:
            return None
        return geojson.Feature(geometry=self.local_maxima.__geo_interface__,
                               properties=dict(lobule_id=self._id))

    @classmethod
    def to_dataframe(cls, statistics: List[LobuleStatistics]) -> pd.DataFrame:
        pass