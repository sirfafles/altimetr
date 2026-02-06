#!/usr/bin/env python3
# fetch_isolines.py
import os
import csv
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple
from shapely.geometry import LineString, Polygon

# ---- helper: same BLN reader as раньше (только для трека) ----
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

# ---- main ----
def read_fetch_csv(path: str) -> Dict[float, float]:
    """
    Читает CSV fetch_point_{idx}.csv, возвращает mapping azimuth_deg -> distance_km (NaN если нет).
    CSV format: azimuth_deg,distance_m,intersect_lon,intersect_lat,reason
    """
    az2km = {}
    if not os.path.exists(path):
        return az2km
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            az = float(row["azimuth_deg"])
            # distance_m column may be empty
            d_m = row.get("distance_m", "").strip()
            if d_m == "" or d_m.lower() in ("no_intersection",):
                az2km[az] = math.nan
            else:
                try:
                    az2km[az] = float(d_m) / 1000.0
                except Exception:
                    az2km[az] = math.nan
    return az2km

def build_grid(indices: List[int], track_file: str, csv_dir: str, out_csv: str
               ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Возвращает (longitudes (n_lon,), azimuths (n_az,), Z (n_az, n_lon))
    и сохраняет grid CSV.
    """
    track = read_bln(track_file, geom_type="line")
    n_points = len(track.coords)

    # собираем долготы для выбранных индексов
    longitudes = []
    for idx in indices:
        if idx < 0 or idx >= n_points:
            raise IndexError(f"Track index {idx} out of range (0..{n_points-1})")
        lon, lat = track.coords[idx]
        longitudes.append(lon)
    # читаем все CSV и собираем полный набор азимутов
    all_azs = set()
    per_point = []
    for idx in indices:
        p = read_fetch_csv(os.path.join(csv_dir, f"fetch_point_{idx}.csv"))
        per_point.append(p)
        all_azs.update(p.keys())
    if not all_azs:
        raise ValueError("No azimuths found in CSV files.")

    azimuths = sorted(all_azs)
    # конструируем Z: строки = azimuths, столбцы = longitudes
    Z = np.full((len(azimuths), len(longitudes)), np.nan, dtype=float)
    az_index = {az:i for i,az in enumerate(azimuths)}
    for j, p in enumerate(per_point):
        for az, km in p.items():
            i = az_index[az]
            Z[i, j] = km

    # сохраняем grid CSV: первая колонка az, потом колонки по долготе
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        header = ["azimuth_deg"] + [f"{lon:.6f}" for lon in longitudes]
        writer.writerow(header)
        for i, az in enumerate(azimuths):
            row = [f"{az:.1f}"] + [("" if math.isnan(Z[i,j]) else f"{Z[i,j]:.6f}") for j in range(Z.shape[1])]
            writer.writerow(row)

    return np.array(longitudes), np.array(azimuths), Z

def plot_isolines(lons: np.ndarray, azs: np.ndarray, Z: np.ndarray, out_png: str):
    # X axis = longitudes, Y axis = azimuths
    X, Y = np.meshgrid(lons, azs)  # shapes (n_az, n_lon)
    Z_masked = np.ma.masked_invalid(Z)

    fig, ax = plt.subplots(figsize=(10,6))
    # Контурные уровни: автоматически либо задать шаг
    # Пропустим plot если всё NaN
    if np.all(np.isnan(Z)):
        ax.text(0.5, 0.5, "No valid data to plot", ha="center", va="center")
    else:
        # выберем уровни по диапазону ненулевых значений
        vmin = np.nanmin(Z)
        vmax = np.nanmax(Z)
        if vmin == vmax:
            levels = 8
        else:
            # 10 уровней равномерно
            levels = np.linspace(vmin, vmax, 10)
        cf = ax.contourf(X, Y, Z_masked, levels=levels, cmap="viridis")
        cbar = fig.colorbar(cf, ax=ax)
        cbar.set_label("Distance (km)")
        # подписать контуры линиями (опционально)
        try:
            cs = ax.contour(X, Y, Z_masked, levels=levels, colors="k", linewidths=0.5)
            ax.clabel(cs, inline=True, fmt="%.1f", fontsize=8)
        except Exception:
            pass

    ax.set_xlabel("Longitude")
    ax.set_ylabel("Azimuth (deg, from north clockwise)")
    ax.set_title("Fetch isolines (km) — X: longitudes of points, Y: azimuths")
    ax.set_xticks(lons := list(lons))
    ax.set_xticklabels([f"{x:.3f}" for x in lons], rotation=45)
    ax.set_ylim(azs.min(), azs.max())
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
    fig.savefig(out_png, dpi=200)
    plt.close(fig)
    print("Saved isolines image to", out_png)

if __name__ == "__main__":
    # параметры — изменяй при необходимости
    csv_dir = "./output"
    track_file = "./data/Track_TP&J1-2_Phase-A_009.bln"
    indices = list(range(10, 17))  # 10..16 включительно
    out_grid_csv = "./output/fetch_isolines_grid.csv"
    out_png = "./output/fetch_isolines.png"

    lons, azs, Z = build_grid(indices, track_file, csv_dir, out_grid_csv)
    print("Longitudes:", lons)
    print("Azimuths:", azs)
    print("Z shape:", Z.shape)
    plot_isolines(lons, azs, Z, out_png)
    print("Grid CSV saved to", out_grid_csv)
