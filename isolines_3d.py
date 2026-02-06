#!/usr/bin/env python3
# fetch_3d_surface.py
"""
Построение интерактивной 3D-поверхности с изолиниями.
Вход: CSV'ки ./output/fetch_point_{idx}.csv или ./output/fetch_isolines_grid.csv
Выход: ./output/fetch_surface.html (интерактивная), ./output/fetch_surface.png (опционально, требует kaleido)
"""
import os
import csv
import math
import numpy as np

# try to import plotly; если не установлен — сообщим и предложим установить
try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
except Exception as e:
    go = None
    _plotly_import_error = e

# --------------------- вспомогательные функции чтения ---------------------
def read_grid_csv(path: str):
    """
    Читает ./output/fetch_isolines_grid.csv в формате:
    header: azimuth_deg, lon1, lon2, ...
    строки: az, val_lon1, val_lon2, ...
    Возвращает (lons(np.array), azs(np.array), Z(np.ndarray shape (n_az, n_lon)) )
    """
    if not os.path.exists(path):
        return None
    azs = []
    lons = None
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        # header[0] == "azimuth_deg", далее заголовки столбцов — долготы
        lons = [float(h) for h in header[1:]]
        for r in reader:
            if not r:
                continue
            azs.append(float(r[0]))
            # остальное могут быть пустые строки
            vals = []
            for v in r[1:]:
                if v is None or v == "" or v.strip() == "":
                    vals.append(np.nan)
                else:
                    try:
                        vals.append(float(v))
                    except:
                        vals.append(np.nan)
            rows.append(vals)
    Z = np.array(rows, dtype=float)  # shape (n_az, n_lon)
    return np.array(lons), np.array(azs), Z

def read_fetch_csv(path: str):
    """
    читает fetch_point_{idx}.csv: azimuth_deg,distance_m,intersect_lon,intersect_lat,reason
    возвращает dict az->distance_km (nan если нет)
    """
    d = {}
    if not os.path.exists(path):
        return d
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            az = float(row["azimuth_deg"])
            d_m = row.get("distance_m", "").strip()
            if d_m == "" or d_m.lower() in ("no_intersection",):
                d[az] = math.nan
            else:
                try:
                    d[az] = float(d_m) / 1000.0
                except:
                    d[az] = math.nan
    return d

def build_grid_from_fetch_points(indices, track_file="./data/Track_TP&J1-2_Phase-A_009.bln", csv_dir="./output"):
    """
    Собирает lons (долготы точек indices), azs, Z (в km) по fetch_point_{idx}.csv
    """
    # читаем трек только чтобы взять долготы (короткая реализация)
    # простой BLN reader:
    def read_bln_line(fn):
        with open(fn, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]
        header = lines[0].replace(",", " ").split()
        n_points = int(header[0])
        coords = []
        for line in lines[1 : n_points + 1]:
            parts = line.replace(",", " ").split()
            lon, lat = map(float, parts[:2])
            coords.append((lon, lat))
        return coords

    coords = read_bln_line(track_file)
    longitudes = []
    for idx in indices:
        if idx < 0 or idx >= len(coords):
            raise IndexError(f"Index {idx} out of range")
        lon, _ = coords[idx]
        longitudes.append(lon)

    # читаем per-point CSVs
    per_point = []
    az_set = set()
    for idx in indices:
        p = read_fetch_csv(os.path.join(csv_dir, f"fetch_point_{idx}.csv"))
        per_point.append(p)
        az_set.update(p.keys())

    azs = sorted(az_set)
    if not azs:
        raise ValueError("Нет данных азимутов в CSV'ах. Убедись, что fetch_point_{idx}.csv существуют в ./output")

    Z = np.full((len(azs), len(longitudes)), np.nan, dtype=float)
    az_index = {az:i for i,az in enumerate(azs)}
    for j, p in enumerate(per_point):
        for az, km in p.items():
            i = az_index[az]
            Z[i, j] = km

    return np.array(longitudes), np.array(azs), Z

# --------------------- основная логика ---------------------
def main():
    out_dir = "./output"
    os.makedirs(out_dir, exist_ok=True)
    grid_csv = os.path.join(out_dir, "fetch_isolines_grid.csv")

    # сначала пробуем прочитать готовый grid CSV
    grid = read_grid_csv(grid_csv)
    if grid is None:
        # fallback: собираем из per-point CSV (индексы 10..16 как у тебя сейчас)
        indices = list(range(10, 17))
        grid = build_grid_from_fetch_points(indices, track_file="./data/Track_TP&J1-2_Phase-A_009.bln", csv_dir=out_dir)

    lons, azs, Z = grid  # Z в км
    print("Loaded grid:", "lons:", lons, "azimuths:", azs, "Z shape:", Z.shape)

    # заменим NaN на None для plotly (json-friendly)
    Z_plot = Z.tolist()
    for i in range(len(Z_plot)):
        for j in range(len(Z_plot[0])):
            if math.isnan(Z_plot[i][j]):
                Z_plot[i][j] = None

    out_html = os.path.join(out_dir, "fetch_surface.html")
    out_png = os.path.join(out_dir, "fetch_surface.png")

    if go is None:
        print("Plotly не установлен. Установи: python -m pip install plotly")
        # можно пока выйти или сгенерировать статический PNG через matplotlib (ниже закомментировано)
        return

    # строим surface: x = longitudes, y = azimuths, z = Z (km)
    # plotly Surface ожидает Z как 2D array shape (len(y), len(x))
    surface = go.Surface(
        x=lons,
        y=azs,
        z=Z_plot,
        colorscale="Viridis",
        colorbar=dict(title="Distance (km)"),
        contours = dict(
            z = dict(show=True, usecolormap=True, highlightcolor="white", project={'z':True})
        ),
        showscale=True
    )

    layout = go.Layout(
        title="Fetch surface (distance, km) — X: longitude of points, Y: azimuth (deg)",
        scene = dict(
            xaxis=dict(title="Longitude"),
            yaxis=dict(title="Azimuth (deg, from north clockwise)"),
            zaxis=dict(title="Distance (km)"),
        ),
        autosize=True,
        margin=dict(l=0, r=0, b=0, t=40)
    )

    fig = go.Figure(data=[surface], layout=layout)

    # добавим подписи координат точек (в заголовке/аннотациях)
    coords_txt = ", ".join([f"{lon:.3f}" for lon in lons])
    fig.update_layout(title=f"Fetch surface — longitudes: {coords_txt}")

    # сохраняем интерактивный HTML
    fig.write_html(out_html, include_plotlyjs="cdn")
    print("Saved interactive surface to", out_html)

    # пробуем также сохранить статический PNG (требует kaleido)
    try:
        fig.write_image(out_png, width=1200, height=800, scale=1)
        print("Saved static PNG to", out_png)
    except Exception as e:
        print("Не удалось сохранить PNG (kaleido может быть не установлен):", e)
        print("Если нужен PNG, установи kaleido: python -m pip install -U kaleido")

if __name__ == "__main__":
    main()
