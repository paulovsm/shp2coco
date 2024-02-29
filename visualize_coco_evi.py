from pycocotools.coco import COCO
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
import pylab
import os
from PIL import Image
from tifffile import imread
from skimage import exposure

ROOT_DIR = r'./example_data/evi_data/dataset/test'
image_directory = os.path.join(ROOT_DIR, "deforestation_2022")
annotation_file = os.path.join(ROOT_DIR, "instances_deforestation_test2022.json")

example_coco = COCO(annotation_file)

category_ids = example_coco.getCatIds(catNms=['square'])
print("category_ids: ", category_ids)
image_ids = example_coco.getImgIds(catIds=category_ids)
print("image_ids: ", image_ids)
image_data = example_coco.loadImgs(image_ids[3])[0]
print("image_data: ", image_data)

print("image path: ", image_directory + '/' + image_data['file_name'])
image_original = io.imread(image_directory + '/' + image_data['file_name'])

print("image_original shape, dtype ", image_original.shape, image_original.dtype)

# Normalizar a imagem para o intervalo [0..1]
image_original = (image_original * 255).astype(np.uint8)

# Esticar o histograma da imagem
image_original = exposure.rescale_intensity(image_original, in_range='image')

# Carregar a imagem
image = io.imread(image_directory + '/' + image_data['file_name'])

# Normalizar a imagem para o intervalo [0..1]
image = (image * 255).astype(np.uint8)

# Esticar o histograma da imagem
image = exposure.rescale_intensity(image, in_range='image')

# Imprimir o tipo de dados da imagem
print('Tipo de dados da imagem:', image.dtype)
# image = io.imread(image_directory + '/' + image_data['file_name'])

# plt.figure()

# # Adicionar a primeira imagem em um subplot
plt.subplot(1, 2, 1) # (nrows, ncols, index)
plt.imshow(image_original, cmap='gray')
plt.axis('off')

# # Adicionar a segunda imagem em outro subplot
plt.subplot(1, 2, 2)
plt.imshow(image, cmap='gray')
plt.axis('off')

# plt.imshow(image); plt.axis('off')
# pylab.rcParams['figure.figsize'] = (8.0, 10.0)
annotation_ids = example_coco.getAnnIds(imgIds=image_data['id'], catIds=category_ids, iscrowd=None)
annotations = example_coco.loadAnns(annotation_ids)
example_coco.showAnns(annotations)

# Salvar copia da imagem original
#io.imsave('output_annon.png', image)

# Salvar a imagem como um arquivo
plt.savefig('output_annon.png')

plt.show()

# Converter a imagem de volta para uint8
#image_original = (image_original * 255).astype(np.uint8)

# Salvar a imagem como PNG
io.imsave('converted_image.png', image_original)