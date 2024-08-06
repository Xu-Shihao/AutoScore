from ultralytics import YOLO
import yaml

config_file = open('./config.yaml')
config = yaml.load(config_file, yaml.loader.SafeLoader)
config_file.close()

model_name = config['model']
weights = config['training_weights']
dataset = config['dataset']

model = YOLO(model=model_name).load(weights=weights)  # build from YAML and transfer weights

model.train(data=dataset, epochs=20, imgsz=640, batch=32, dropout=0.1, lr0 = 1e-2, lrf=1e-6)
