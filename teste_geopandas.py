import geopandas as gpd
import rasterio
from rasterio.plot import show
from rasterio.features import geometry_mask
import matplotlib.pyplot as plt
import numpy as np
from rasterio.plot import show
import skimage.io as io
from skimage import exposure

# Load the TIFF image using rasterio
tiff_path = './example_data/test_data/img/2017RGB432.tif'
with rasterio.open(tiff_path) as src:
    image = src.read(1)

src = rasterio.open(tiff_path)

# Load the shapefile using geopandas
shapefile_path = './example_data/test_data/shp/2017RGB432/2017RGB432.shp'
gdf = gpd.read_file(shapefile_path)

# Create a binary mask from the shapefile
mask = geometry_mask(gdf.geometry, out_shape=image.shape, transform=src.transform, invert=True)

# print numpy array pixel values count
unique, counts = np.unique(mask, return_counts=True)
print(dict(zip(unique, counts)))

# Overlay the mask with the image
overlay_image = image.copy()
overlay_image[mask] = 255  # Set the masked pixels to a white color (assuming 8-bit image)

# # Display the overlay
# fig = plt.subplots()
# show(overlay_image, cmap="gray")
# # Save the binary image
# output_image_path = './example_data/test_data/output/mask_image.png'
# plt.savefig(output_image_path, format='png', dpi=300)

# Transpose the image array to correct the shape
# image_display = np.transpose(image, (1, 2, 0))
# print(image_display.dtype)

# # Normalizar a imagem para o intervalo [0..1]
# image_display = image_display / 65535.0

# # # Adicionar a primeira imagem em um subplot
# plt.subplot(1, 2, 1) # (nrows, ncols, index)
# plt.imshow(image_display)
# plt.axis('off')

# # # Adicionar a segunda imagem em outro subplot
# plt.subplot(1, 2, 2)
# plt.imshow(mask)
# plt.axis('off')

fig, (axr, axg, axb) = plt.subplots(1,3, figsize=(21,7))

show((src, 1), ax=axr, cmap='Reds', title='red channel')
show((src, 2), ax=axg, cmap='Greens', title='green channel')
show((src, 3), ax=axb, cmap='Blues', title='blue channel')

# Save the binary image
output_image_path = './example_data/test_data/output/rgb_bands.png'
plt.savefig(output_image_path, format='png', dpi=300)


# Carregar a imagem
image_io = io.imread(tiff_path)

print("image_io shape, dtype ", image_io.shape, image_io.dtype)

# Normalizar a imagem para o intervalo [0..1]
image_io = image_io / 65535.0

# Esticar o histograma da imagem
image_io = exposure.rescale_intensity(image_io, in_range='image')

plt.figure()
plt.imshow(image_io)
plt.axis('off')

# Save the binary image
output_image_path = './example_data/test_data/output/mask_image.png'
plt.savefig(output_image_path, format='png', dpi=300)