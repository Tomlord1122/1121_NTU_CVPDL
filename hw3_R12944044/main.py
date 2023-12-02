from transformers import Blip2Processor, Blip2ForConditionalGeneration
import torch
from PIL import Image
import json


device = "cuda" if torch.cuda.is_available() else "cpu"

# load the pre-trained model
# processor = Blip2Processor.from_pretrained(
#     "Salesforce/blip2-opt-2.7b")  # 處理data
# model = Blip2ForConditionalGeneration.from_pretrained(  # 用於生成圖像的字敘述
#     "Salesforce/blip2-opt-2.7b", torch_dtype=torch.float16)

processor = Blip2Processor.from_pretrained('./model')
model = Blip2ForConditionalGeneration.from_pretrained(
    './model', torch_dtype=torch.float16)


model.to(device)

# Image Path

selected_img = [
    "IMG_8406_jpg.rf.fda4b68f345bda8047e7f15060f70e45.jpg",	"IMG_3147_jpeg_jpg.rf.fc4622004ff72e58b546635774372fe2.jpg",	"IMG_2402_jpeg_jpg.rf.ff2e5af0a2d1693c155a01d7494fc8e4.jpg",	"IMG_2584_jpeg_jpg.rf.fdf498ef5b1b000a6c51d15dc29ad33a.jpg",
    "IMG_2489_jpeg_jpg.rf.ffb357957a29cdef43f3fdfb2a13c417.jpg",	"IMG_2481_jpeg_jpg.rf.00a2836323b67c925752c28bccc26ea4.jpg",	"IMG_8595_MOV-2_jpg.rf.055891706f32e5310829f3a0a46cec5e.jpg",	"IMG_8599_MOV-2_jpg.rf.0b2b0733befaae0b08c0e04b86f295b9.jpg",
    "IMG_2335_jpeg_jpg.rf.fee7aabbe3a95b58fa737bf4537ded6f.jpg",	"IMG_3169_jpeg_jpg.rf.00b985cb8ddf5da8964dc21435e1bd2c.jpg",	"IMG_2309_jpeg_jpg.rf.088f73ff0b07c30ce6212eb4e3013708.jpg",	"IMG_2303_jpeg_jpg.rf.0c70be2d073ae94feeb463b79babaab8.jpg",
    "IMG_3131_jpeg_jpg.rf.fcf0ccd8dbf187344da242e29eeac0cb.jpg",	"IMG_3138_jpeg_jpg.rf.ff253449ce146d664f1c0fb5f7f25ef5.jpg",	"IMG_3158_jpeg_jpg.rf.0f2bac58bc2e3ca07172215f073a2dab.jpg",	"IMG_2299_jpeg_jpg.rf.19c4728e89a506c9a7dcbb2509d6a134.jpg",
    "IMG_2509_jpeg_jpg.rf.0c8d6158f08975bd24497a5fb02572d2.jpg",	"IMG_8493_jpg.rf.0f25b61a8d7ab12cbf5ae131582007d5.jpg",	"IMG_2560_jpeg_jpg.rf.121b55027c132565ca11f89e21ea6722.jpg",	"IMG_2597_jpeg_jpg.rf.0f07eb126fe35d31acdbec7adb23ed50.jpg",
    "IMG_3121_jpeg_jpg.rf.0868c7d1a4e32a0ca167c247eefea800.jpg",	"IMG_2542_jpeg_jpg.rf.1986fa80da6bc8506a8dc1c918b82b03.jpg",	"IMG_3127_jpeg_jpg.rf.1f29f97225daf147f37da45ccc3e9f5e.jpg",	"IMG_2386_jpeg_jpg.rf.24ef35f21ff0b936ede4bb3a50c0886e.jpg",
    "IMG_2613_jpeg_jpg.rf.03a46725a60afa2d492d463de939b335.jpg",	"IMG_2625_jpeg_jpg.rf.049cfca5fa7da90878060904a452975d.jpg",	"IMG_2589_jpeg_jpg.rf.0796c64369ea29d7e50fc1f7a36fb25b.jpg",	"IMG_2617_jpeg_jpg.rf.0749a7644179240c7747ad03ae9999c0.jpg"
]

caption_res = []
for img in selected_img:
    # Load the image
    image = Image.open(f"./hw1_dataset/train/{img}")  # use PIL load the image

    # Use Proceesor to process image => 轉換成model可以理解的格式
    inputs = processor(images=image, return_tensors="pt").to(device)

    # generate caption from model.
    outputs = model.generate(**inputs, max_new_tokens=40)
    generated_caption = processor.batch_decode(
        outputs, skip_special_tokens=True)[0]
    print("Generated Caption:", generated_caption)


with open("output_val.json", "w") as json_file:
    json.dump(caption_res, json_file, indent=4)


model.save_pretrained("./model")
processor.save_pretrained('./model')
