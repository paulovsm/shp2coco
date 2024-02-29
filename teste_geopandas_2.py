import geopandas as gpd
import rasterio
from rasterio.plot import show
from rasterio.features import geometry_mask
import matplotlib.pyplot as plt

# Load the TIFF image using rasterio
tiff_path = './example_data/test_data/img/2017RGB432.tif'
with rasterio.open(tiff_path) as src:
    image = src.read(1)  # Assuming a single-band image for simplicity

# Load the shapefile using geopandas
shapefile_path = './example_data/test_data/shp/2017RGB432/2017RGB432.shp'
gdf = gpd.read_file(shapefile_path)

# Create a binary mask from the shapefile
mask = geometry_mask(gdf.geometry, out_shape=image.shape, transform=src.transform, invert=True)

# Create a binary image
fig, ax = plt.subplots()
ax.set_aspect('equal')
gdf.plot(ax=ax, color='black', edgecolor='black')

# Save the binary image
output_image_path = './example_data/test_data/output/mask_image.png'
plt.savefig(output_image_path, format='png', dpi=300)
