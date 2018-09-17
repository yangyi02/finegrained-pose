# StanfordCars 3D train
CUDA_VISIBLE_DEVICES=1 python compare_annotation.py \
--image_dir=../Image/StanfordCars/cars_train \
--model_dir=../CAD/02958343 \
--anno_file=../Anno3D/StanfordCars/train_anno.pkl \
--new_anno_dir=../Anno3D_v2/StanfordCars/train_anno
--overlay_dir=../Overlay/StanfordCars/cars_train \
--new_overlay_dir=../Overlay_v2/StanfordCars/cars_train

# StanfordCars 3D test
CUDA_VISIBLE_DEVICES=1 python compare_annotation.py \
--image_dir=../Image/StanfordCars/cars_test \
--model_dir=../CAD/02958343 \
--anno_file=../Anno3D/StanfordCars/test_anno.pkl \
--new_anno_dir=../Anno3D_v2/StanfordCars/test_anno \
--overlay_dir=../Overlay/StanfordCars/cars_test \
--new_overlay_dir=../Overlay_v2/StanfordCars/cars_test

# FGVC-Aircraft 3D train
CUDA_VISIBLE_DEVICES=1 python compare_annotation.py \
--image_dir ../Image/FGVC_Aircraft \
--model_dir ../CAD/02691156 \
--anno_file ../Anno3D/FGVC_Aircraft/train_anno.pkl \
--new_anno_dir ../Anno3D_v2/FGVC_Aircraft/train_anno \
--overlay_dir=../Overlay/FGVC_Aircraft \
--new_overlay_dir=../Overlay_v2/FGVC_Aircraft
 
# FGVC-Aircraft 3D test
CUDA_VISIBLE_DEVICES=1 python compare_annotation.py \
--image_dir ../Image/FGVC_Aircraft \
--model_dir ../CAD/02691156 \
--anno_file ../Anno3D/FGVC_Aircraft/test_anno.pkl \
--new_anno_dir ../Anno3D_v2/FGVC_Aircraft/test_anno \
--overlay_dir=../Overlay/FGVC_Aircraft \
--new_overlay_dir=../Overlay_v2/FGVC_Aircraft
