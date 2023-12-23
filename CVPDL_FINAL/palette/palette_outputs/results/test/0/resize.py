# want to resize the image 
from PIL import Image



def resize_image(image_path, output_path, new_width, new_height):
    image = Image.open(image_path)
    image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    image.save(output_path)
    
resize_image('Out_image1.png', 'Out_image1.png', 810, 1080)
resize_image('Out_image2.png', 'Out_image2.png', 810, 1080)
resize_image('Out_image3.png', 'Out_image3.png', 810, 1080)
    
