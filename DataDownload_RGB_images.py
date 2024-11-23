import openeo
import rasterio
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import unary_union
import xarray as xr
import imageio


# Connessione al backend di OpenEO
connection = openeo.connect("openeo.vito.be").authenticate_oidc(provider_id='terrascope')

# Definizione delle coordinate per l'area di Torino
spatial_extent = {
    "west": 7.58,
    "south": 45.03,
    "east": 7.72,
    "north": 45.13,
}
temporal_extent = ["2023-09-01", "2023-09-30"]
# Carica le bande per l'immagine RGB
bands = ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"]
collection = connection.load_collection(
    "TERRASCOPE_S2_TOC_V2",
    spatial_extent=spatial_extent,
    temporal_extent=temporal_extent,
    bands = bands,
)

collection.download('torino_rgb.nc', format='netCDF')
print("Download completato: torino_rgb.nc")
dataset = xr.open_dataset('torino_rgb.nc')
print("Date disponibili:", dataset['t'].values)
# open the original data
date = "2023-09-11"
original = xr.open_dataset('torino_rgb.nc').sel(t=date)[bands].to_array().values
original = original[:, :original.shape[1], :original.shape[2]]
rgb_original = np.clip(original[[2,1,0]]/4000,0,1).transpose(1,2,0)
# Salva l'immagine RGB come PNG
imageio.imwrite('rgb_original.png', (rgb_original * 255).astype('uint8'))

plt.figure(figsize=(10, 10))  # Imposta la dimensione del plot
plt.imshow(rgb_original, interpolation='nearest')
plt.title("Immagine RGB Originale", fontsize=16)
plt.axis("off")  # Nascondi gli assi
plt.show()