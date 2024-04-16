import types
from xml.etree.ElementTree import Element, SubElement, tostring
import os
from exif import Image
from PIL import Image as ImageMod
from transformers import BlipProcessor, BlipForConditionalGeneration
import xml.dom.minidom

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
        with open(os.path.join(images_directory, imageFile),'rb') as image_file:
            image = Image(image_file)

            photo = SubElement(root, "photo")
            photo.set("id", str(id))
            photo.set("name", imageFileName)

            exif_node = SubElement(photo, "EXIF")
            description = SubElement(photo, "description")

            id += 1
            print(id)
            # add exif
            for key in dir(image):
                if key != '_segments':
                    val = image.get(key)
                    if not hasattr(val, '__self__'):
                        exif = SubElement(exif_node, key)
                        exif.text = str(val)


            raw_image = ImageMod.open(os.path.join(images_directory, imageFile)).convert('RGB')

            # unconditional image captioning
            inputs = processor(raw_image, return_tensors="pt")

            out = model.generate(**inputs)

            description.text = processor.decode(out[0], skip_special_tokens=True)


with open('xml.xml','w') as file:
    file.write(xml.dom.minidom.parseString(tostring(root)).toprettyxml())
