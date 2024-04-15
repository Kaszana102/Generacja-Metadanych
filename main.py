from exif import Image
from xml.etree.ElementTree import Element, SubElement, tostring
import os
import requests
from PIL import Image as ImageMod
from transformers import BlipProcessor, BlipForConditionalGeneration

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
root = Element('root')

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

images_directory = os.fsencode('images')

id = 0

for imageFile in os.listdir(images_directory):
    imageFileName = os.fsdecode(imageFile)
    if imageFileName.endswith(".jpg"):
        # print(os.path.join(directory, filename))
        with open(os.path.join(images_directory, imageFile), 'rb') as image_handle:

            photo = SubElement(root, "photo")
            photo.set("id", str(id))
            photo.set("name", imageFileName)
            exif = SubElement(photo, "EXIF")
            description = SubElement(root, "description")
            description.text = "opis"
            id += 1

            image = Image(image_handle)
            if image.has_exif:
                dir(image)
                # add exif
                for trait in dir(image):
                    if image.get(trait) is not None:
                        ex = SubElement(exif, trait)
                        ex.text = str(image.get(trait))

            raw_image = ImageMod.open(os.path.join(images_directory, imageFile)).convert('RGB')

            # unconditional image captioning
            inputs = processor(raw_image, return_tensors="pt")

            out = model.generate(**inputs)
            print(processor.decode(out[0], skip_special_tokens=True))

#print(tostring(root))
