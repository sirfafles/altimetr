#!/usr/bin/env python3
# script.py
import matplotlib
import pyproj
from shapely.geometry import LineString, Point, Polygon

matplotlib.use("Agg")
import csv
import os
from typing import List

import matplotlib.pyplot as plt


def read_bln(filename: str, geom_type: str = "polygon"):
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
    else:
        raise ValueError(f"Unknown geometry type: {geom_type}")


def extract_points_from_geometry(geom) -> List[Point]:
    pts = []
    if geom.is_empty:
        return pts
    if geom.geom_type == "Point":
        pts.append(geom)
    elif geom.geom_type in (
        "MultiPoint",
        "GeometryCollection",
        "MultiLineString",
        "MultiPolygon",
    ):
        for g in geom.geoms:
            pts.extend(extract_points_from_geometry(g))
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


def calculate_north_fetchs(lake_polygon: Polygon, track_line: LineString) -> List[dict]:
    geod = pyproj.Geod(ellps="WGS84")
    results = []
    lake_minx, lake_miny, lake_maxx, lake_maxy = lake_polygon.bounds
    lake_boundary = lake_polygon.boundary

    for idx, (lon, lat) in enumerate(track_line.coords):
        if lat >= lake_maxy:
            results.append(
                {
                    "index": idx,
                    "track_point": (lon, lat),
                    "intersection_point": None,
                    "distance_m": None,
                    "reason": "point_north_of_lake",
                }
            )
            continue

        # строим луч до выше максимальной широты озера + запас
        ray_end_lat = max(lat + 0.01, lake_maxy + 0.05)  # 0.05° запас ~5-6 km
        ray = LineString([(lon, lat), (lon, ray_end_lat)])

        inter = lake_boundary.intersection(ray)
        candidates = extract_points_from_geometry(inter)
        # фильтруем строго северные
        candidates_north = [p for p in candidates if p.y > lat + 1e-12]

        if not candidates_north:
            results.append(
                {
                    "index": idx,
                    "track_point": (lon, lat),
                    "intersection_point": None,
                    "distance_m": None,
                    "reason": "no_intersection",
                }
            )
            continue

        # выберем ближайшую северную точку по геодезическому расстоянию
        best = None
        best_d = None
        for p in candidates_north:
            _, _, dist_m = geod.inv(lon, lat, p.x, p.y)
            if best is None or dist_m < best_d:
                best = p
                best_d = dist_m

        results.append(
            {
                "index": idx,
                "track_point": (lon, lat),
                "intersection_point": (best.x, best.y),
                "distance_m": best_d,
                "reason": "ok",
            }
        )

    return results


def save_csv(results, out_csv="./output/fetches.csv"):
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "index",
                "lon",
                "lat",
                "intersect_lon",
                "intersect_lat",
                "distance_m",
                "reason",
            ]
        )
        for r in results:
            idx = r["index"]
            lon, lat = r["track_point"]
            if r["intersection_point"] is None:
                writer.writerow(
                    [idx, f"{lon:.6f}", f"{lat:.6f}", "", "", "", r["reason"]]
                )
            else:
                ilon, ilat = r["intersection_point"]
                writer.writerow(
                    [
                        idx,
                        f"{lon:.6f}",
                        f"{lat:.6f}",
                        f"{ilon:.6f}",
                        f"{ilat:.6f}",
                        f"{r['distance_m']:.2f}",
                        r["reason"],
                    ]
                )
    print("Saved CSV to", out_csv)


def visualize(
    lake_polygon: Polygon,
    track_line: LineString,
    results,
    out_png="./output/fetches.png",
):
    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 8))

    # lake
    x, y = lake_polygon.exterior.xy
    ax.plot(x, y, linewidth=1.0, label="Lake boundary")

    # track
    tx, ty = track_line.xy
    ax.plot(tx, ty, linewidth=1.0, label="Track", color="red")

    # points and rays
    for r in results:
        lon, lat = r["track_point"]
        ax.scatter(lon, lat, s=8, color="red")
        if r["intersection_point"] is not None:
            ilon, ilat = r["intersection_point"]
            ax.plot([lon, ilon], [lat, ilat], linestyle="--", linewidth=0.8)
            ax.scatter(ilon, ilat, s=10, color="blue")

    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("Fetch (northward) from track points to lake boundary")
    ax.legend()
    fig.tight_layout()
    plt.savefig(out_png, dpi=200)
    print("Saved visualization to", out_png)
    plt.close(fig)


def main():
    lake_polygon = read_bln(
        "./data/Lake_Peipus_HydroLAKES_polys_v10.bln", geom_type="polygon"
    )
    print("Lake:", lake_polygon.is_valid, lake_polygon.area)

    track_line = read_bln("./data/Track_TP&J1-2_Phase-A_009.bln", geom_type="line")
    print("Track:", track_line.is_valid, track_line.length)

    results = calculate_north_fetchs(lake_polygon, track_line)
    save_csv(results)
    visualize(lake_polygon, track_line, results)

    print("index, lon, lat, intersect_lon, intersect_lat, distance_m, reason")
    for r in results:
        idx = r["index"]
        lon, lat = r["track_point"]
        if r["intersection_point"] is None:
            print(f"{idx}, {lon:.6f}, {lat:.6f}, , , , {r['reason']}")
        else:
            ilon, ilat = r["intersection_point"]
            print(
                f"{idx}, {lon:.6f}, {lat:.6f}, {ilon:.6f}, {ilat:.6f}, {r['distance_m']:.2f}, {r['reason']}"
            )


if __name__ == "__main__":
    main()
