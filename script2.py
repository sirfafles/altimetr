#!/usr/bin/env python3
# fetch_angles.py
import os
import csv
from typing import List, Tuple, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyproj
from shapely.geometry import LineString, Point, Polygon

# ---- вставь сюда функцию read_bln и extract_points_from_geometry (как в твоём коде) ----

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

# ---- основная логика ----

def compute_max_distance_to_lake_vertices(geod: pyproj.Geod, lon: float, lat: float, lake_polygon: Polygon) -> float:
    """
    Вернёт максимальное геодезическое расстояние (в метрах) от точки (lon,lat)
    до вершин полигона озера. Затем добавим буфер.
    """
    max_d = 0.0
    for vx, vy in lake_polygon.exterior.coords:
        _, _, d = geod.inv(lon, lat, vx, vy)
        if d > max_d:
            max_d = d
    return max_d

def calculate_fetchs_for_point(lake_polygon: Polygon, point_coord: Tuple[float,float],
                               azimuths: List[float], geod: pyproj.Geod) -> List[dict]:
    """
    Для данной точки и списка азимутов возвращает список словарей:
    { "az":deg, "distance_m": float|None, "intersect": (lon,lat)|None, "reason": str }
    """
    lon0, lat0 = point_coord
    lake_boundary = lake_polygon.boundary

    # выставим длину луча = max distance до вершин + запас (10%)
    max_d = compute_max_distance_to_lake_vertices(geod, lon0, lat0, lake_polygon)
    ray_length = max(max_d * 1.2, 1000.0)  # минимум 1 km, иначе 20% от max_d

    results = []
    for az in azimuths:
        # получаем конечную точку по азимуту и расстоянию ray_length
        # geod.fwd принимает lon, lat, azimuth_deg, distance_m -> возвращает (lon2, lat2, backaz)
        lon_end, lat_end, _ = geod.fwd(lon0, lat0, az, ray_length)
        ray = LineString([(lon0, lat0), (lon_end, lat_end)])

        inter = lake_boundary.intersection(ray)
        candidates = extract_points_from_geometry(inter)
        # оставляем только точки (вдоль сегмента) — shapely даст точки именно на сегменте,
        # поэтому они уже в "переднем" направлении; всё же проверим,
        # что расстояние > 0 (не нулевая точка)
        candidates = [p for p in candidates if (abs(p.x - lon0) > 1e-12 or abs(p.y - lat0) > 1e-12)]

        if not candidates:
            results.append({"az": az, "distance_m": None, "intersect": None, "reason": "no_intersection"})
            continue

        # у кандидатов может быть несколько точек (например берег извилист).
        # выбираем ближайшую по геодезическому расстоянию.
        best = None
        best_d = None
        best_p = None
        for p in candidates:
            _, _, d = geod.inv(lon0, lat0, p.x, p.y)
            # чтобы быть уверенными, что точка действительно в "передней" половине, проверяем азимут
            # от стартовой точки к кандидатной: он должен быть близок к az (±90°) — но
            # т.к. мы строим сегмент вперед, это, как правило, выполняется.
            if best is None or d < best_d:
                best = p
                best_d = d
                best_p = p
        if best_p is None:
            results.append({"az": az, "distance_m": None, "intersect": None, "reason": "no_valid_candidate"})
        else:
            results.append({"az": az, "distance_m": float(best_d), "intersect": (best_p.x, best_p.y), "reason": "ok"})
    return results

def save_point_results_csv(results: List[dict], out_csv: str):
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["azimuth_deg", "distance_m", "intersect_lon", "intersect_lat", "reason"])
        for r in results:
            if r["intersect"] is None:
                writer.writerow([f"{r['az']:.1f}", "" , "", "", r["reason"]])
            else:
                ilon, ilat = r["intersect"]
                writer.writerow([f"{r['az']:.1f}", f"{r['distance_m']:.3f}", f"{ilon:.6f}", f"{ilat:.6f}", r["reason"]])

def plot_point_results(results: List[dict], point_coord: Tuple[float,float], out_png: str, title_coords_fmt: str):
    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
    azs = [r["az"] for r in results]
    dists = [r["distance_m"] if r["distance_m"] is not None else np.nan for r in results]

    fig, ax = plt.subplots(figsize=(10,5))
    ax.plot(azs, dists, marker='o', linestyle='-')
    ax.set_xticks(azs)
    ax.set_xlabel("Azimuth (deg) — from north, clockwise")
    ax.set_ylabel("Distance (m)")
    ax.set_title(f"Fetch vs Azimuth — point {title_coords_fmt}")
    ax.grid(True)
    fig.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close(fig)

# ---- main ----

def main():
    data_dir = "./data"
    out_dir = "./output"
    os.makedirs(out_dir, exist_ok=True)

    lake_polygon = read_bln(os.path.join(data_dir, "Lake_Peipus_HydroLAKES_polys_v10.bln"), geom_type="polygon")
    track_line = read_bln(os.path.join(data_dir, "Track_TP&J1-2_Phase-A_009.bln"), geom_type="line")
    print("Lake:", lake_polygon.is_valid, "area (deg^2):", lake_polygon.area)
    print("Track:", track_line.is_valid, "n_points:", len(track_line.coords))

    # индексы точек, которые просим (11..17 включительно)
    indices = list(range(10, 17))
    geod = pyproj.Geod(ellps="WGS84")
    azimuths = list(range(0, 360, 10))  # 0,10,...350

    for idx in indices:
        if idx < 0 or idx >= len(track_line.coords):
            print(f"Index {idx} out of range, skipping.")
            continue
        lon, lat = track_line.coords[idx]
        print(f"Processing point index {idx} -> lon={lon:.6f}, lat={lat:.6f}")

        results = calculate_fetchs_for_point(lake_polygon, (lon, lat), azimuths, geod)

        csv_path = os.path.join(out_dir, f"fetch_point_{idx}.csv")
        png_path = os.path.join(out_dir, f"fetch_point_{idx}.png")

        save_point_results_csv(results, csv_path)
        title_coords = f"{lon:.6f}, {lat:.6f}"
        plot_point_results(results, (lon, lat), png_path, title_coords)

        print(f"Saved CSV: {csv_path}")
        print(f"Saved PNG: {png_path}")

    print("Done.")

if __name__ == "__main__":
    main()
