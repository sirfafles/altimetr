#!/usr/bin/env python3
"""
Скрипт для расчёта разгона (fetch): расстояние от точки внутри озера до границы
озера в заданном направлении (азимут в градусах).

Вход: координаты озера (файл BLN или полигон), координаты точки (lon, lat), угол (азимут, 0° = север, 90° = восток).
Выход: расстояние в метрах и точка пересечения с берегом.
"""

import argparse
from typing import List, Optional, Tuple

import pyproj
from shapely.geometry import LineString, Point, Polygon


def read_bln(filename: str, geom_type: str = "polygon") -> Polygon:
    """Читает полигон или линию из BLN-файла."""
    with open(filename, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]
    header = lines[0].replace(",", " ").split()
    n_points = int(header[0])

    coords = []
    for line in lines[1 : n_points + 1]:
        parts = line.replace(",", " ").split()
        lon, lat = map(float, parts[:2])
        coords.append((lon, lat))

    if geom_type == "polygon":
        return Polygon(coords)
    elif geom_type == "line":
        return LineString(coords)
    raise ValueError(f"Unknown geometry type: {geom_type}")


def _extract_points_from_geometry(geom) -> List[Point]:
    """Собирает все точки из результата intersection (Point, MultiPoint и т.д.)."""
    pts = []
    if geom.is_empty:
        return pts
    if geom.geom_type == "Point":
        pts.append(geom)
    elif geom.geom_type in ("MultiPoint", "GeometryCollection", "MultiLineString", "MultiPolygon"):
        for g in geom.geoms:
            pts.extend(_extract_points_from_geometry(g))
    elif geom.geom_type in ("LineString", "LinearRing"):
        for c in geom.coords:
            pts.append(Point(c))
    else:
        try:
            for c in geom.coords:
                pts.append(Point(c))
        except Exception:
            pass
    return pts


def _max_distance_to_boundary(geod: pyproj.Geod, lon: float, lat: float, polygon: Polygon) -> float:
    """Максимальное геодезическое расстояние от точки до вершин полигона (м)."""
    max_d = 0.0
    for x, y in polygon.exterior.coords:
        _, _, d = geod.inv(lon, lat, x, y)
        if d > max_d:
            max_d = d
    return max_d


def fetch_distance(
    lake_polygon: Polygon,
    point_lon: float,
    point_lat: float,
    angle_deg: float,
    geod: Optional[pyproj.Geod] = None,
) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """
    Разгон: расстояние от точки до границы озера в направлении angle_deg.

    Азимут: 0° = север, 90° = восток (по часовой стрелке).

    Возвращает:
        (distance_m, intersect_lon, intersect_lat) при успехе;
        (None, None, None) если пересечения нет или точка не внутри озера.
    """
    if geod is None:
        geod = pyproj.Geod(ellps="WGS84")

    if not lake_polygon.contains(Point(point_lon, point_lat)):
        return (None, None, None)

    boundary = lake_polygon.boundary
    max_d = _max_distance_to_boundary(geod, point_lon, point_lat, lake_polygon)
    ray_length = max(max_d * 1.2, 1000.0)

    lon_end, lat_end, _ = geod.fwd(point_lon, point_lat, angle_deg, ray_length)
    ray = LineString([(point_lon, point_lat), (lon_end, lat_end)])

    inter = boundary.intersection(ray)
    candidates = _extract_points_from_geometry(inter)
    # исключаем саму стартовую точку
    candidates = [
        p
        for p in candidates
        if (abs(p.x - point_lon) > 1e-12 or abs(p.y - point_lat) > 1e-12)
    ]

    if not candidates:
        return (None, None, None)

    best_p = None
    best_d = None
    for p in candidates:
        _, _, d = geod.inv(point_lon, point_lat, p.x, p.y)
        if best_d is None or d < best_d:
            best_d = d
            best_p = p

    if best_p is None:
        return (None, None, None)
    return (float(best_d), best_p.x, best_p.y)


def main():
    parser = argparse.ArgumentParser(
        description="Разгон: расстояние от точки внутри озера до берега по заданному азимуту."
    )
    parser.add_argument(
        "lake_bln",
        help="Путь к BLN-файлу с полигоном озера (первая строка: N,0; далее N строк lon,lat).",
    )
    parser.add_argument("lon", type=float, help="Долгота точки внутри озера (градусы).")
    parser.add_argument("lat", type=float, help="Широта точки внутри озера (градусы).")
    parser.add_argument(
        "angle",
        type=float,
        help="Азимут направления (градусы): 0 = север, 90 = восток.",
    )
    parser.add_argument(
        "--km",
        action="store_true",
        help="Выводить расстояние в километрах.",
    )
    args = parser.parse_args()

    lake = read_bln(args.lake_bln, geom_type="polygon")
    dist_m, i_lon, i_lat = fetch_distance(lake, args.lon, args.lat, args.angle)

    if dist_m is None:
        print("Пересечения с границей не найдено (или точка вне озера).")
        return 1

    if args.km:
        print(f"Разгон: {dist_m / 1000:.6f} km")
    else:
        print(f"Разгон: {dist_m:.3f} m")
    if i_lon is not None and i_lat is not None:
        print(f"Точка пересечения с берегом: {i_lon:.6f}, {i_lat:.6f}")
    return 0


if __name__ == "__main__":
    exit(main())
