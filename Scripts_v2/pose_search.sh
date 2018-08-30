# StanfordCars 3D train
python pose_search.py \
--image_dir=../Image/StanfordCars/cars_train \
--model_dir=../CAD/02958343 \
--anno_file=../Anno3D/StanfordCars/train_anno.pkl \
--deeplab_model_path=./deeplab_model/deeplabv3_pascal_trainval_2018_01_04.tar.gz \
--maskrcnn_model_path=./maskrcnn_model/mask_rcnn_coco.h5 \
--class_label=car \
--new_anno_dir=../Anno3D/StanfordCars/train_anno_new

# StanfordCars 3D test
python pose_search.py \
--image_dir=../Image/StanfordCars/cars_test \
--model_dir=../CAD/02958343 \
--anno_file=../Anno3D/StanfordCars/test_anno.pkl \
--deeplab_model_path=./deeplab_model/deeplabv3_pascal_trainval_2018_01_04.tar.gz \
--maskrcnn_model_path=./maskrcnn_model/mask_rcnn_coco.h5 \
--class_label=car \
--new_anno_dir=../Anno3D/StanfordCars/test_anno_new

# FGVC-Aircraft 3D train
python pose_search.py \
--image_dir ../Image/FGVC_Aircraft \
--model_dir ../CAD/02691156 \
--anno_file ../Anno3D/FGVC_Aircraft/train_anno.pkl \
--deeplab_model_path=./deeplab_model/deeplabv3_pascal_trainval_2018_01_04.tar.gz \
--maskrcnn_model_path=./maskrcnn_model/mask_rcnn_coco.h5 \
--class_label=aeroplane \
--new_anno_dir ../Anno3D/FGVC_Aircraft/train_anno_new

# FGVC-Aircraft 3D test
python pose_search.py \
--image_dir ../Image/FGVC_Aircraft \
--model_dir ../CAD/02691156 \
--anno_file ../Anno3D/FGVC_Aircraft/test_anno.pkl \
--deeplab_model_path=./deeplab_model/deeplabv3_pascal_trainval_2018_01_04.tar.gz \
--maskrcnn_model_path=./maskrcnn_model/mask_rcnn_coco.h5 \
--class_label=aeroplane \
--new_anno_dir ../Anno3D/FGVC_Aircraft/test_anno_new
