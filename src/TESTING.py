import src.data_utils.coco_to_yolo as cocoyo
from src.data_utils.coco_to_yolo import make_yaml_file

# make_yaml_file("dataset", "yaml_dataset") 
cocoyo.convert_coco_to_yolo("data/train_annots_modify", "data/train_labels_YOLO")

# from src.train_YOLO import train_YOLO
# train_YOLO("data/train_images/train", "data/train_labels/train", batch_size=8, num_epochs=5, lr=0.001, weight_decay=0.005, optimizer_name="sgd", scheduler_name="step", device="cpu", debug=False)


