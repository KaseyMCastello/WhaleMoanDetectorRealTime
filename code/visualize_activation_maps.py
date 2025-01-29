import torch
import cv2
import torchvision
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from AudioDetectionDataset import AudioDetectionData_with_hard_negatives
from pytorch_grad_cam import EigenCAM
from pytorch_grad_cam.utils.model_targets import FasterRCNNBoxScoreTarget
from pytorch_grad_cam.utils.reshape_transforms import fasterrcnn_reshape_transform
from pytorch_grad_cam.utils.image import show_cam_on_image
import numpy as np
from PIL import Image

# Assuming you have a device setup (e.g., CUDA or CPU)
# Path to the trained model
model_path = "L:/WhaleMoanDetector/models/WhaleMoanDetector.pth"  # Replace with the correct path

# Load the pre-trained Faster R-CNN model
model = torchvision.models.detection.fasterrcnn_resnet50_fpn()

# Define the number of classes (5 classes + 1 background)
num_classes = 6  # Adjust this based on your training setup

# Get the input features for the classifier (this is the number of features coming from the backbone)
in_features = model.roi_heads.box_predictor.cls_score.in_features

# Replace the classifier head with a new one (for your dataset)
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

# Load the trained weights from your model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.load_state_dict(torch.load(model_path, map_location=device))  # Load the model weights
model.to(device)
model.eval()  # Set the model to evaluation mode

# Prepare DataLoader
val_d1 = DataLoader(AudioDetectionData_with_hard_negatives(csv_file='../labeled_data/train_val_test_annotations/val.csv'),
                    batch_size=1,  # Usually batch size is 1 during inference
                    shuffle=False, 
                    collate_fn=None)  # Adjust if you have a custom collate function

# Initialize the CAM model (EigenCAM)
target_layers = [model.backbone]  # Typically use the FPN backbone
cam = EigenCAM(model=model, 
               target_layers=target_layers,
               reshape_transform=fasterrcnn_reshape_transform)

# Run inference and visualize the activation maps for predictions
for data in val_d1:
    
    # Visualize the predictions with bounding boxes
    img = data[0][0].to(device)  # Image tensor
    print("input image tensor shape:", img.shape)  # Should print (1, 141, 601)
    
    # Ensure the image tensor has the correct shape: [1, 3, H, W]
    img = img.repeat(3, 1, 1)  # Repeat the grayscale image to 3 channels
    
    # Verify the shape after ensuring 3 channels
    print("input image tensor after shape adjustment:", img.shape)  # Should print (1, 3, 141, 601)
    
 
    
    # Run inference on the image
    with torch.no_grad():
        output = model(img)[0]  # Model output
    
    # Extract predicted boxes, labels, and scores
    boxes = output['boxes'].cpu().numpy()
    labels = output['labels'].cpu().numpy()
    scores = output['scores'].cpu().numpy()
    
    # Filter predictions by a threshold score (optional)
    threshold = 0.5
    indices = np.where(scores >= threshold)[0]
    boxes = boxes[indices]
    labels = labels[indices]
    
    # Convert tensor to numpy array and transpose to [H, W, C] format for visualization
    image_np = img[0].cpu().numpy()  # Convert from [C, H, W] to [H, W, C]
    
    # Normalize to [0, 1] (if necessary)
    image_np = image_np / image_np.max()  # Ensure the pixel values are between 0 and 1
    
    # Draw bounding boxes on the image
    for box in boxes:
        color = np.random.rand(3,)  # Random color for each box
        image_np = cv2.rectangle(image_np, 
                                 (int(box[0]), int(box[1])), 
                                 (int(box[2]), int(box[3])), 
                                 color.tolist(), 2)
    
    # Generate CAM for each detected object using EigenCAM
    targets = [FasterRCNNBoxScoreTarget(labels=torch.tensor(labels), bounding_boxes=torch.tensor(boxes))]
    grads = cam(input_tensor=img, targets=targets)
    
    # Overlay CAM on image
    cam_image = show_cam_on_image(image_np, grads[0, 0, :, :], use_rgb=True)
    
    # Display the result
    plt.imshow(cam_image)
    plt.axis('off')
    plt.show()
