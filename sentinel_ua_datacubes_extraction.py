import geopandas as gpd
from shapely.geometry import box
import openeo
import os
import imageio
from datetime import datetime, timedelta


# 1. Load the shapefile
def load_shapefile(shapefile_path):
    gdf = gpd.read_file(shapefile_path)
    print("Shapefile CRS:", gdf.crs)
    return gdf


# 2. Split the ROI in grids of the desired boundaries
def create_grid(roi, lat_increment=0.0285, lon_increment=0.0426):
    minx, miny, maxx, maxy = roi.total_bounds
    x_coords = list(frange(minx, maxx, lon_increment))
    y_coords = list(frange(miny, maxy, lat_increment))
    grid = []
    for x in x_coords:
        for y in y_coords:
            cell = box(x, y, x + lon_increment, y + lat_increment)
            if roi.intersects(cell).any():  # Controlla che il box intersechi il ROI
                grid.append(cell)
    grid_gdf = gpd.GeoDataFrame({'geometry': grid}, crs=roi.crs)
    print(f"Total cells: {len(grid_gdf)}")
    return grid_gdf


def frange(start, stop, step):
    while start < stop:
        yield start
        start += step


# 3. Download the sentinel data and create the datacubes
def download_satellite_images(geometry, start_date, end_date, connection, collection='SENTINEL2_L2A'):
    print(f"Downloading data for collection: {collection}")
    datacube = connection.load_collection(
        collection_id=collection,
        spatial_extent={
            "west": geometry.bounds[0],
            "south": geometry.bounds[1],
            "east": geometry.bounds[2],
            "north": geometry.bounds[3],
        },
        temporal_extent=[start_date, end_date],
        bands=["B04", "B08"]  # Bande necessarie per NDVI o visualizzazione
    )
    return datacube


# 4. Create a timelapse
def create_timelapse(image_folder, output_path):
    images = [imageio.imread(os.path.join(image_folder, file))
              for file in sorted(os.listdir(image_folder)) if file.endswith(".png")]
    if not images:
        print(f"No images found in {image_folder}, skipping timelapse creation.")
        return
    imageio.mimsave(output_path, images, fps=2)


# 5. Main
def main(shapefile_path, start_date, end_date, output_folder):
    connection = openeo.connect("https://openeo.cloud")
    connection.authenticate_oidc()

    roi = load_shapefile(shapefile_path)
    grid = create_grid(roi)
    print("Grid CRS:", grid.crs)
    print("Grid bounds:", grid.total_bounds)
    print("Shapefile bounds:", roi.total_bounds)

    # Filter the cells and intersect with ROI
    filtered_grid = grid[grid.geometry.apply(lambda cell: roi.intersects(cell).any())]
    print(f"Filtered cells count: {len(filtered_grid)}")

    for i, row in filtered_grid.iterrows():
        cell = row.geometry
        print(f"Processing bounding box: {cell.bounds}")
        print(f"Processing cell {i + 1}/{len(filtered_grid)}")

        # Download the satellite data
        datacube = download_satellite_images(cell, start_date, end_date, connection)

        try:
            # Execute job batch for image download
            job = datacube.create_job()
            job.start_and_wait()

            # Obtain results and download
            results = job.get_results()
            result_folder = os.path.join(output_folder, f"cell_{i + 1}")
            os.makedirs(result_folder, exist_ok=True)
            results.download_files(target=result_folder)

            # Verification of downloaded images
            downloaded_files = os.listdir(result_folder)
            if not downloaded_files:
                print(f"No data downloaded for cell {i + 1}, skipping timelapse creation.")
                continue

            # Timelapse creation
            output_timelapse_path = os.path.join(output_folder, f"timelapse_cell_{i + 1}.mp4")
            create_timelapse(result_folder, output_timelapse_path)
            print(f"Timelapse saved at {output_timelapse_path}")

        except openeo.rest.JobFailedException as e:
            print(f"Job failed for cell {i + 1}: {e}")
            continue


if __name__ == "__main__":
    shapefile_path = "shapefile_ua_donetsk/2021 HNO GCA_NGCA.shp"  # Path to the shapefile
    start_date = '2023-01-01'  # Beginning date (YYYY-MM-DD)
    end_date = '2023-12-31'    # Final date
    output_folder = "output"   # Output folder

    main(shapefile_path, start_date, end_date, output_folder)
