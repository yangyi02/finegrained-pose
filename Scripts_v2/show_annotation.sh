# Visualize the first training image of the three dataset

# StanfordCars 3D
python show_annotation.py \
--image_dir ../Image/StanfordCars/cars_train \
--model_dir ../CAD/02958343 \
--anno_file ../Anno3D/StanfordCars/train_anno.pkl \
--draw_statistics

# StanfordCars 3D v2
python show_annotation.py \
--image_dir ../Image/StanfordCars/cars_train \
--model_dir ../CAD/02958343 \
--anno_file ../Anno3D/StanfordCars/train_anno_v2.pkl \
--draw_statistics

# CompCars 3D
python show_annotation.py \
--image_dir ../Image/CompCars/image \
--model_dir ../CAD/02958343 \
--anno_file ../Anno3D/CompCars/train_anno.pkl \
--draw_statistics

# FGVC-Aircraft 3D
python show_annotation.py \
--image_dir ../Image/FGVC_Aircraft/ \
--model_dir ../CAD/02691156 \
--anno_file ../Anno3D/FGVC_Aircraft/train_anno.pkl \
--draw_statistics

# FGVC-Aircraft 3D
python show_annotation.py \
--image_dir ../Image/FGVC_Aircraft/ \
--model_dir ../CAD/02691156 \
--anno_file ../Anno3D/FGVC_Aircraft/train_anno_v2.pkl \
--draw_statistics
