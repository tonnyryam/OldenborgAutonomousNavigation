import time

from PIL import Image

from jetbot import Camera

# wait 5 seconds before first image to give time to put camera in right position
time.sleep(5)

camera = Camera()

for i in range(500):
    image_arr = camera.value
    print(i)
    image = Image.fromarray(image_arr)
    image.save("all_images/" + str(i) + ".png")
    time.sleep(1)
