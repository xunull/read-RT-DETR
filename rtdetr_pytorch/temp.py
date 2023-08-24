import json
from collections import defaultdict

new_file = '/mnt/h/ml_dataset_home/coco/annotations/instances_train2017_remove_1021.json'

with open('/mnt/h/ml_dataset_home/coco/annotations/instances_train2017.json') as file:
    data = json.load(file)
    origin = data.copy()
    images = data['images']
    annos = data['annotations']
    no_anno = []
    id_anno = defaultdict(list)
    for anno in annos:
        id = anno['image_id']
        id_anno[id].append(anno)

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

    with open(new_file, 'w') as newf:
        json.dump(data, newf, indent=4)

    print("新文件制作完成")
