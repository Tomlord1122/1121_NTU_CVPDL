from ultralytics import YOLO
import numpy as np
import cv2
import torch

model = YOLO('yolov8n-seg.pt', task='segment')  # Load a pretrained model

# Load Image
img = cv2.imread('/home/tomlord/Desktop/ultralytics/ultralytics/assets/bus.jpg')
# imgPath = './ultralytics/assets/'  # Path to image, yolov8 can predict multiple images at once
# save img
# cv2.imwrite('./ultralytics/assets/image2.png', img)
h, w, _ = img.shape  # Get the original height and width of the image

# Perform prediction
results = model.predict(source=img,
                        conf = 0.5,
                        save = True,
                        save_txt = False,
                        retina_masks = True, # use retina masks
                        classes = 0, # only people
                        # save_crop = True,
                        # show_labels = True,
                        # show_conf = False,
                        # show_boxes = True,
                        )



# Process the results
for i, result in enumerate(results):
    # Get array results
    masks = result.masks.data
    boxes = result.boxes.data
    # Extract classes
    clss = boxes[:, 5]
    # Get indices of results where class is 0 (people in COCO)
    people_indices = torch.where(clss == 0)[0]
    # Use these indices to extract the relevant masks
    people_masks = masks[people_indices]
    # Combine masks for all people detected
    combined_mask = torch.any(people_masks, dim=0).int()
    # Scale mask to original image size
    scaled_mask = cv2.resize(combined_mask.cpu().numpy(), (w, h), interpolation=cv2.INTER_NEAREST) * 255
    
    
    
    
    
    
    for idx in people_indices:
        # 提取特定人物的 mask
        person_mask = masks[idx]
        # 將 mask 轉換為二值影像
        binary_mask = (person_mask > 0).int()
        # Scale mask to original image size
        scaled_mask = cv2.resize(binary_mask.cpu().numpy(), (w, h), interpolation=cv2.INTER_NEAREST) * 255
        
        # save to PNG file
        save_path = str(model.predictor.save_dir / f'image{i+1}_{idx}.png')
        cv2.imwrite(save_path, scaled_mask)
    
    
    
    
    
    # Save to file in PNG format
    # save_path = str(model.predictor.save_dir / f'image{i+1}.png')
    # cv2.imwrite(save_path, scaled_mask)