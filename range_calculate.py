from shapely.geometry import LineString, Polygon


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


lake_polygon = read_bln(
    "./data/Lake_Peipus_HydroLAKES_polys_v10.bln", geom_type="polygon"
)
print("Lake:", lake_polygon.is_valid, lake_polygon.area)

track_line = read_bln("./data/Track_TP&J1-2_Phase-A_009.bln", geom_type="line")
print("Track:", track_line.is_valid, track_line.length)
