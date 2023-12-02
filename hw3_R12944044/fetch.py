import json

selected_img = [
    "IMG_8406_jpg.rf.fda4b68f345bda8047e7f15060f70e45.jpg",	"IMG_3147_jpeg_jpg.rf.fc4622004ff72e58b546635774372fe2.jpg",	"IMG_2402_jpeg_jpg.rf.ff2e5af0a2d1693c155a01d7494fc8e4.jpg",	"IMG_2584_jpeg_jpg.rf.fdf498ef5b1b000a6c51d15dc29ad33a.jpg",
    "IMG_2489_jpeg_jpg.rf.ffb357957a29cdef43f3fdfb2a13c417.jpg",	"IMG_2481_jpeg_jpg.rf.00a2836323b67c925752c28bccc26ea4.jpg",	"IMG_8595_MOV-2_jpg.rf.055891706f32e5310829f3a0a46cec5e.jpg",	"IMG_8599_MOV-2_jpg.rf.0b2b0733befaae0b08c0e04b86f295b9.jpg",
    "IMG_2335_jpeg_jpg.rf.fee7aabbe3a95b58fa737bf4537ded6f.jpg",	"IMG_3169_jpeg_jpg.rf.00b985cb8ddf5da8964dc21435e1bd2c.jpg",	"IMG_2309_jpeg_jpg.rf.088f73ff0b07c30ce6212eb4e3013708.jpg",	"IMG_2303_jpeg_jpg.rf.0c70be2d073ae94feeb463b79babaab8.jpg",
    "IMG_3131_jpeg_jpg.rf.fcf0ccd8dbf187344da242e29eeac0cb.jpg",	"IMG_3138_jpeg_jpg.rf.ff253449ce146d664f1c0fb5f7f25ef5.jpg",	"IMG_3158_jpeg_jpg.rf.0f2bac58bc2e3ca07172215f073a2dab.jpg",	"IMG_2299_jpeg_jpg.rf.19c4728e89a506c9a7dcbb2509d6a134.jpg",
    "IMG_2509_jpeg_jpg.rf.0c8d6158f08975bd24497a5fb02572d2.jpg",	"IMG_8493_jpg.rf.0f25b61a8d7ab12cbf5ae131582007d5.jpg",	"IMG_2560_jpeg_jpg.rf.121b55027c132565ca11f89e21ea6722.jpg",	"IMG_2597_jpeg_jpg.rf.0f07eb126fe35d31acdbec7adb23ed50.jpg",
    "IMG_3121_jpeg_jpg.rf.0868c7d1a4e32a0ca167c247eefea800.jpg",	"IMG_2542_jpeg_jpg.rf.1986fa80da6bc8506a8dc1c918b82b03.jpg",	"IMG_3127_jpeg_jpg.rf.1f29f97225daf147f37da45ccc3e9f5e.jpg",	"IMG_2386_jpeg_jpg.rf.24ef35f21ff0b936ede4bb3a50c0886e.jpg",
    "IMG_2613_jpeg_jpg.rf.03a46725a60afa2d492d463de939b335.jpg",	"IMG_2625_jpeg_jpg.rf.049cfca5fa7da90878060904a452975d.jpg",	"IMG_2589_jpeg_jpg.rf.0796c64369ea29d7e50fc1f7a36fb25b.jpg",	"IMG_2617_jpeg_jpg.rf.0749a7644179240c7747ad03ae9999c0.jpg"
]

description = [

    "a large aquarium with fish swimming in it",
    "a penguin is swimming in the water",
    "two fish swimming in an aquarium with coral",
    "a large aquarium with fish and coral",
    "jellyfishs in the aquarium",
    "jellyfishs swimming in an aquarium with blue water",
    "many jellyfishs are swimming in an aquarium",
    "a group of jellyfish floating in the ocean",
    "penguins in the water at the zoo",
    "penguins in the water at the zoo",
    "a fish swimming in the water",
    "a penguin swimming in an aquarium with rocks",
    "puffin flying over the ocean",
    "puffin flying over the ocean",
    "puffin flying over the ocean",
    "puffin flying over the ocean",
    "a shark is swimming in an aquarium",
    "a fish is swimming in an aquarium",
    "a group of sharks swimming in an aquarium",
    "a large aquarium with a shark and a rock",
    "a starfish and a fish in a tank",
    "a bunch of blue and green sea stars",
    "a fish tank with rocks and a sea anemone",
    "starfish in the aquarium",
    "a shark is swimming in an aquarium",
    "a white stingray in an aquarium tank",
    "a large aquarium with fish and coral",
    "a fish is in a tank"

]

file_path = "./hw1_dataset/annotations/train.json"


# 透過 ImageName找到其對應的ImageId, height, width.

with open(file_path, 'r', encoding='utf-8') as file:
    data_structure = json.load(file)
image_data = data_structure.get("images", [])
image_id = dict()
image_heightInfo = dict()
image_widthInfo = dict()
image_bboxInfo = dict()

for imgName in selected_img:
    for image in image_data:
        if image.get("file_name") == imgName:
            image_id[imgName] = image.get("id")
            image_heightInfo[imgName] = image.get("height")
            image_widthInfo[imgName] = image.get("width")

# 透過ImageId 找到在Annotations中對應的bbox

annotations_data = data_structure.get(
    "annotations", []) if data_structure else []


for imgName in selected_img:
    bboxes = []
    for annotation in annotations_data:
        if annotation.get("image_id") == image_id[imgName]:
            bbox = annotation.get("bbox")
            if bbox:
                x_min = bbox[0]
                y_min = bbox[1]
                x_max = bbox[0] + bbox[2]  # x + width
                y_max = bbox[1] + bbox[3]  # y + height

                img_width = image_widthInfo[imgName]
                img_height = image_heightInfo[imgName]

                # normalize axis
                x_min_normalized = round(x_min / img_width, 2)
                y_min_normalized = round(y_min / img_height, 2)
                x_max_normalized = round(x_max / img_width, 2)
                y_max_normalized = round(y_max / img_height, 2)

                normalized_bbox = [
                    x_min_normalized, y_min_normalized, x_max_normalized, y_max_normalized]
                bboxes.append(normalized_bbox)
    image_bboxInfo[imgName] = bboxes


# use selected_img fetch imgName

finalOutput = {}


for index, imgName in enumerate(selected_img):
    finalOutput[imgName] = {
        "image": imgName,
        "image_id": image_id.get(imgName),
        "height": image_heightInfo.get(imgName),
        "width": image_widthInfo.get(imgName),
        "bbox": image_bboxInfo.get(imgName),
        "generated_text": description[index]

    }


with open("output_prompt.json", "w") as json_file:
    json.dump(finalOutput, json_file, indent=4)
