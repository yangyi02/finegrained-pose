import os
import pickle as pkl

anno_file = '../Anno3D/StanfordCars/train_anno.pkl'
new_anno_dir = '../Anno3D/StanfordCars/train_anno_new'
new_anno_file = '../Anno3D/StanfordCars/train_anno_new.pkl'

with open(anno_file, 'rb') as f:
    annos = pkl.load(f)

keys = sorted(annos.keys())
for key in keys:
    # load new annotation
    image_id, ext = os.path.splitext(key)
    anno_file = image_id + '.pkl'
    with open(os.path.join(new_anno_dir, anno_file), 'rb') as f:
        new_annos = pkl.load(f)
    new_anno = new_annos[key]
    annos[key] = new_anno

with open(new_anno_file, 'wb') as handle:
    pkl.dump(annos, handle)
