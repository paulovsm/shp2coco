from pycocotools.coco import COCO
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
import pylab
import os
from PIL import Image
from tifffile import imread

ROOT_DIR = r'./example_data/original_data/dataset/eval'
image_directory = os.path.join(ROOT_DIR, "greenhouse_2019")
annotation_file = os.path.join(ROOT_DIR, "instances_greenhouse_eval2019.json")

example_coco = COCO(annotation_file)

category_ids = example_coco.getCatIds(catNms=['square'])
print("category_ids: ", category_ids)
image_ids = example_coco.getImgIds(catIds=category_ids)
print("image_ids: ", image_ids)
image_data = example_coco.loadImgs(image_ids[3])[0]
print("image_data: ", image_data)

print("image path: ", image_directory + '/' + image_data['file_name'])
image_original = io.imread(image_directory + '/' + image_data['file_name'])

# Converter a imagem para uint8
image_original = image_original.astype(np.uint8)

# Carregar a imagem
image = io.imread(image_directory + '/' + image_data['file_name'])

# Converter a imagem para uint8
image = image.astype(np.uint8)

# Imprimir o tipo de dados da imagem
print('Tipo de dados da imagem:', image.dtype)
# image = io.imread(image_directory + '/' + image_data['file_name'])

# plt.figure()

# # Adicionar a primeira imagem em um subplot
plt.subplot(1, 2, 1) # (nrows, ncols, index)
plt.imshow(image_original)
plt.axis('off')

# # Adicionar a segunda imagem em outro subplot
plt.subplot(1, 2, 2)
plt.imshow(image)
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