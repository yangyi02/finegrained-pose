# Visualize the first training image of the three dataset

# # StanfordCars 3D
# python extract_deeplab_mask.py \
# --image_dir ../Image/StanfordCars/cars_train \
# --model_path ../deeplab_model/deeplabv3_pascal_trainval_2018_01_04.tar.gz \
# --segment_dir ../Segment_DeepLab/StanfordCars/cars_train \
# --class_label car \
# --visualize
# 
# python3 extract_mrcnn_mask.py \
# --image_dir ../Image/StanfordCars/cars_train \
# --model_path ../maskrcnn_model/mask_rcnn_coco.h5 \
# --segment_dir ../Segment_MaskRCNN/StanfordCars/cars_train \
# --class_label car \
# --visualize
# 
# python3 extract_final_mask.py \
# --image_dir ../Image/StanfordCars/cars_train \
# --anno_file ../Anno3D/StanfordCars/train_anno.pkl \
# --model_dir ../CAD/02958343 \
# --deeplab_segment_dir ../Segment_DeepLab/StanfordCars/cars_train \
# --mrcnn_segment_dir ../Segment_MaskRCNN/StanfordCars/cars_train \
# --segment_dir ../Segment_Final/StanfordCars/cars_train \
# --class_label car \
# --visualize
# 
# python pose_search.py \
# --image_dir ../Image/StanfordCars/cars_train \
# --model_dir ../CAD/02958343 \
# --anno_file ../Anno3D/StanfordCars/train_anno.pkl \
# --segment_dir ../Segment_Final/StanfordCars/cars_train \
# --new_anno_dir ../Anno3D/StanfordCars/train_anno_new

# # StanfordCars 3D
# python extract_deeplab_mask.py \
# --image_dir ../Image/StanfordCars/cars_test \
# --model_path ../deeplab_model/deeplabv3_pascal_trainval_2018_01_04.tar.gz \
# --segment_dir ../Segment_DeepLab/StanfordCars/cars_test \
# --class_label car \
# --visualize

# python3 extract_mrcnn_mask.py \
# --image_dir ../Image/StanfordCars/cars_test \
# --model_path ../maskrcnn_model/mask_rcnn_coco.h5 \
# --segment_dir ../Segment_MaskRCNN/StanfordCars/cars_test \
# --class_label car \
# --visualize

# python3 extract_final_mask.py \
# --image_dir ../Image/StanfordCars/cars_test \
# --anno_file ../Anno3D/StanfordCars/test_anno.pkl \
# --model_dir ../CAD/02958343 \
# --deeplab_segment_dir ../Segment_DeepLab/StanfordCars/cars_test \
# --mrcnn_segment_dir ../Segment_MaskRCNN/StanfordCars/cars_test \
# --segment_dir ../Segment_Final/StanfordCars/cars_test \
# --class_label car \
# --visualize

python pose_search.py \
--image_dir ../Image/StanfordCars/cars_test \
--model_dir ../CAD/02958343 \
--anno_file ../Anno3D/StanfordCars/test_anno.pkl \
--segment_dir ../Segment_Final/StanfordCars/cars_test \
--new_anno_dir ../Anno3D/StanfordCars/test_anno_new

# FGVC-Aircraft 3D
python extract_deeplab_mask.py \
--image_dir ../Image/FGVC_Aircraft \
--model_path ../deeplab_model/deeplabv3_pascal_trainval_2018_01_04.tar.gz \
--segment_dir ../Segment_DeepLab/FGVC_Aircraft \
--class_label aeroplane \
--visualize

python3 extract_mrcnn_mask.py \
--image_dir ../Image/FGVC_Aircraft \
--model_path ../maskrcnn_model/mask_rcnn_coco.h5 \
--segment_dir ../Segment_MaskRCNN/FGVC_Aircraft \
--class_label aeroplane \
--visualize

python3 extract_final_mask.py \
--image_dir ../Image/FGVC_Aircraft \
--anno_file ../Anno3D/FGVC_Aircraft/train_anno.pkl \
--model_dir ../CAD/02691156 \
--deeplab_segment_dir ../Segment_DeepLab/FGVC_Aircraft \
--mrcnn_segment_dir ../Segment_MaskRCNN/FGVC_Aircraft \
--segment_dir ../Segment_Final/FGVC_Aircraft \
--class_label aeroplane \
--visualize

python pose_search.py \
--image_dir ../Image/FGVC_Aircraft \
--model_dir ../CAD/02691156 \
--anno_file ../Anno3D/FGVC_Aircraft/train_anno.pkl \
--segment_dir ../Segment_Final/FGVC_Aircraft \
--new_anno_dir ../Anno3D/FGVC_Aircraft/train_anno_new

# FGVC-Aircraft 3D
python3 extract_final_mask.py \
--image_dir ../Image/FGVC_Aircraft \
--anno_file ../Anno3D/FGVC_Aircraft/test_anno.pkl \
--model_dir ../CAD/02691156 \
--deeplab_segment_dir ../Segment_DeepLab/FGVC_Aircraft \
--mrcnn_segment_dir ../Segment_MaskRCNN/FGVC_Aircraft \
--segment_dir ../Segment_Final/FGVC_Aircraft \
--class_label aeroplane \
--visualize

python pose_search.py \
--image_dir ../Image/FGVC_Aircraft \
--model_dir ../CAD/02691156 \
--anno_file ../Anno3D/FGVC_Aircraft/test_anno.pkl \
--segment_dir ../Segment_Final/FGVC_Aircraft \
--new_anno_dir ../Anno3D/FGVC_Aircraft/test_anno_new
