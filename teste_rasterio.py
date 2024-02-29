# Importar as bibliotecas
import rasterio
import geopandas as gpd
import skimage.io as io
from rasterio.mask import mask
import matplotlib.pyplot as plt
import numpy as np
from skimage.color import rgb2gray
from skimage import exposure
import matplotlib.colors as mcolors

# Carregar o shapefile e o raster de entrada
shapefile = gpd.read_file('./example_data/test_data/shp/2017RGB432/2017RGB432.shp')
raster = rasterio.open('./example_data/test_data/img/2017RGB432.tif')

# Imprimir as primeiras linhas do GeoDataFrame
print(shapefile.head())

# Imprimir o sistema de referência de coordenadas
print("CRS:", shapefile.crs)

# Imprimir as coordenadas do primeiro polígono
print("Coordinates:", shapefile.geometry[0])

#shapefile = gpd.read_file('./example_data/original_data/shp/21/21.shp')
#raster = rasterio.open('./example_data/original_data/img/21.tif')

# Filtrar o GeoDataFrame para incluir apenas a categoria desejada
shapefile_filtered = shapefile[shapefile['CLASSIFIC'] == 1]

# Imprimir as primeiras linhas do GeoDataFrame
print(shapefile_filtered.head())

# Obter a geometria do shapefile filtrado
geometry = shapefile_filtered.geometry

# Criar uma máscara binária a partir da geometria
mask = rasterio.mask.mask(raster, geometry, crop=False, invert=False)

# Obter o dicionário de metadados do raster de entrada
meta = raster.meta

# Atualizar o dicionário com a nova forma e o novo transform da máscara
meta.update({"height": mask[0].shape[1], "width": mask[0].shape[2], "transform": mask[1]})

# Salvar a máscara como um novo raster
with rasterio.open('./example_data/test_data/output/mask.tif', "w", **meta) as dst:
    dst.write(mask[0])

#Carregar a imagem
imagem_io = io.imread('./example_data/test_data/img/2017RGB432.tif')
mask_io = io.imread('./example_data/test_data/output/mask.tif')

print("mask_io shape, dtype ", mask_io.shape, mask_io.dtype)

# # Normalizar a imagem para o intervalo [0..1]
imagem_io = imagem_io / 65535.0

# # Esticar o histograma da imagem
imagem_io = exposure.rescale_intensity(imagem_io, in_range='image')

# Converter todos os pixel com valor diferente de zero para 1
mask_io = np.where(mask_io > 0, 255, 0).astype(np.uint8)

plt.figure()
plt.imshow(mask_io)
plt.axis('off')

# Save the binary image
output_image_path = './example_data/test_data/output/mask_image_tif.png'
plt.savefig(output_image_path, format='png', dpi=300)

# Criar uma figura e um eixo
fig, ax = plt.subplots()

# Converter a imagem RGB para escala de cinza
mask_io_gray = rgb2gray(mask_io)

# Aplicar um limiar para obter uma imagem binária
mask_io_binary = np.where(mask_io_gray > 0, 1, 0)

# print numpy array pixel values count
unique, counts = np.unique(mask_io_binary, return_counts=True)
print(dict(zip(unique, counts)))

# Mostrar a imagem original no eixo
ax.imshow(imagem_io)

# Create a custom colormap
cmap = mcolors.ListedColormap(['none', 'purple'])

# Mostrar a máscara binária com transparência vermelha no mesmo eixo
# O parâmetro alpha controla a transparência da máscara
# O parâmetro cmap define o mapa de cores da máscara
# O parâmetro zorder define a ordem de sobreposição da máscara sobre a imagem
ax.imshow(mask_io_binary, alpha=0.3, cmap=cmap, zorder=1)

# Remover os eixos da figura
ax.axis('off')

# Save the binary image
output_image_path = './example_data/test_data/output/mask_image_overlay.png'
plt.savefig(output_image_path, format='png', dpi=300)