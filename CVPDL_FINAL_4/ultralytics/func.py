from ultralytics import YOLO
import numpy as np
import cv2
import torch
def func(imagePath:str):
    # Load a model
    model = YOLO('yolov8n-seg.pt', task='segment')  # Load a pretrained model

    # Load Image
    img = cv2.imread(imagePath)
    # imgPath = './ultralytics/assets/'  # Path to image, yolov8 can predict multiple images at once
    # save img
    # cv2.imwrite('./ultralytics/assets/image2.png', img)
    h, w, _ = img.shape  # Get the original height and width of the image
    cv2.imwrite("../edge-connect/examples/tom/images/image1.png", img) # store to edge-connect
    cv2.imwrite("../palette/images/image1.png", img) 
   # Perform prediction
    results = model.predict(source=img,
                            conf=0.5,
                            save=True,
                            save_txt=False,
                            retina_masks=True,  # use retina masks
                            classes=0,  # only people
                            )

    # copy原图用于绘制
    annotated_img = img.copy()

    # Process the results
    for i, result in enumerate(results):
        masks = result.masks.data
        boxes = result.boxes.data
        clss = boxes[:, 5]

        # Get indices of results where class is 0 (people in COCO)
        people_indices = torch.where(clss == 0)[0]
        
        for idx, person_idx in enumerate(people_indices):
            person_mask = masks[person_idx]
            #print("person_mask: ",person_mask,"person_idx: ",person_idx, "idx: ",idx)

            # 将 mask 转换为二值图像
            binary_mask = (person_mask > 0).int()

            # 找到 mask 的轮廓
            contours, _ = cv2.findContours(binary_mask.cpu().numpy().astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            # print(len(contours))
            # 取 mask 的第一个轮廓（假设每个人只有一个连续的 mask 区域）
            # for cnt in contours:
            # 计算轮廓的矩
            M = cv2.moments(contours[-1])
            # 检查除数是否为零
            if M['m00'] != 0:
                cx = int(M['m10']/M['m00'])
                cy = int(M['m01']/M['m00'])
                # 在中心点绘制编号
                cv2.putText(annotated_img, str(idx+1), (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                print("enter")
                #print("Person {} is at ({}, {})".format(idx+1, cx, cy))
            else:
                # 如果 M['m00'] 是 0, 不能计算质心，可以选择跳过或者采取其他措施
                pass
            
            
            # 提取特定人物的 mask
            person_mask = masks[idx]
            # 將 mask 轉換為二值影像
            binary_mask = (person_mask > 0).int()
            # Scale mask to original image size
            scaled_mask = cv2.resize(binary_mask.cpu().numpy(), (w, h), interpolation=cv2.INTER_NEAREST) * 255
            
            # save to PNG file
            save_path = str(model.predictor.save_dir / f'image{i+1}_{idx}.png')
            save_path2 = str(f'../edge-connect/examples/tom/maskCol/image{i+1}_{idx}.png') # store to edge-connect
            save_path3 = str(f'../palette/maskCol/image{i+1}_{idx}.png') # store to edge-connect
            
            
            cv2.imwrite(save_path, scaled_mask)
            cv2.imwrite(save_path2, scaled_mask)
            cv2.imwrite(save_path3, scaled_mask)
                
    # 显示带编号的图像
    cv2.imshow("Annotated Image", annotated_img)
    # cv2.waitKey(0)
    cv2.destroyAllWindows()

    # 保存带编号的图像
    cv2.imwrite('annotated_image.png', annotated_img)
    
    return "annotated_image.png"
