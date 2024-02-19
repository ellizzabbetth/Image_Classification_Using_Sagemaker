import subprocess
subprocess.call(['pip', 'install', 'smdebug'])

import smdebug
import json
import logging
import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import io
import requests

# https://knowledge.udacity.com/questions/944605
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))
JSON_CONTENT_TYPE = 'application/json'
JPEG_CONTENT_TYPE = 'image/jpeg'
ACCEPTED_CONTENT_TYPE = [ JPEG_CONTENT_TYPE ]


def net():
    model = models.resnet50(pretrained = True)

    for param in model.parameters():
        param.requires_grad = False   

    num_features = model.fc.in_features
    model.fc = nn.Sequential(nn.Linear(num_features, 512),
                             nn.Dropout(p = 0.2),
                             nn.ReLU(inplace = True),
                             nn.Linear(512, 256),
                             nn.Dropout(p = 0.2),
                             #nn.ReLU(inplace = True),
                             #nn.Dropout(p = 0.2),
                             #nn.Linear(512, 256),
                             nn.ReLU(inplace = True),
                             nn.Linear(256, 133)
                            )
    return model


def model_fn(model_dir):
    logger.info("In model_fn. Model directory is -", model_dir)
    
    
    #model = models.resnet50(pretrained = True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = net()
    
    with open(os.path.join(model_dir, "model.pth"), "rb") as f:
        logger.info("Loading the dog-classifier model.")
        logger.info("Model_dir content: ")
        logger.info(os.listdir(model_dir))
        checkpoint = torch.load(f, map_location = device)
        model.load_state_dict(checkpoint)
        #model.load_state_dict(torch.load(f, map_location = device))
        logger.info('Model loaded successfully')
    
    model.eval()
    model.to(device)
    
    return model

def input_fn(request_body, content_type):
    # Process an image uploaded to the endpoint
    logger.info(f'Content type: {content_type}')
    logger.info(f'Request body type: {type(request_body)}')
    if content_type == 'image/jpeg':
        return Image.open(io.BytesIO(request_body))
    raise Exception('Requested unsupported ContentType in content_type: {}'.format(content_type))

def predict_fn(input_object, model):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info("Predicting...")
    transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
    input_object = transform(input_object)
    
    if torch.cuda.is_available():
        input_object = input_object.cuda() #put data into GPU
    logger.info("Transforms ready")
    model.eval()
    with torch.no_grad():
        prediction = model(input_object.unsqueeze(0))
    return prediction

# postprocess
def output_fn(predictions, content_type):
    assert content_type == "application/json"
    res = predictions.cpu().numpy().tolist()
    return json.dumps(res)

# Serialize the prediction result into the desired response content type
def output_fn(prediction, accept=JSON_CONTENT_TYPE):        
    logger.info('Serializing the generated output.')
    if accept == JSON_CONTENT_TYPE: 
        logger.debug(f'Returning response {json.dumps(prediction)}')
        return json.dumps(prediction), accept
    raise Exception('Requested unsupported ContentType in Accept: {}'.format(accept))