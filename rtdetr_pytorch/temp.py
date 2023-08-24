import json
from collections import defaultdict

new_file = '/mnt/h/ml_dataset_home/coco/annotations/instances_train2017_remove_1021.json'

with open('/mnt/h/ml_dataset_home/coco/annotations/instances_train2017.json') as file:
    data = json.load(file)
    origin = data.copy()
    images = data['images']
    print("原始images 数量 {}".format(len(images)))
    annos = data['annotations']
    no_anno = []
    id_anno = defaultdict(list)
    for anno in annos:
        id = anno['image_id']
        id_anno[id].append(anno)
        bbox = anno.get('bbox', None)
        if bbox is None:
            print("anno {} no bbox".format(anno['id']))
        else:
            if len(bbox) == 0:
                print("anno {} no bbox".format(anno['id']))


    valid_image = []
    invalid_image_id = []
    for image in images:
        id = image['id']
        if id in id_anno:
            valid_image.append(image)
        else:
            invalid_image_id.append(id)
    print("invalid images: {}".format(len(invalid_image_id)))

    data['images'] = valid_image

    print(len(data['images']))

    # with open(new_file, 'w') as newf:
    #     json.dump(data, newf, indent=4)

    print("新文件制作完成")

# Loaded 118287 images in COCO format from /mnt/h/ml_dataset_home/coco/annotations/instances_train2017.json
# Removed 1021 images with no usable annotations. 117266 images left.
