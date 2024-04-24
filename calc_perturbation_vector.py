from PIL import Image, ImageChops

image_path1 = 'results/run1/images/99.png'
image_path2 = 'results/run1/images/0.png'

#Subtract one image from another.
img1 = Image.open(image_path1)
img2 = Image.open(image_path2)
assert(img1.size == img2.size)

subtracted_image =  ImageChops.subtract(img1, img2)

#Scale image pixel values so the maximum pixel value becomes 255.
pixels = subtracted_image.load()
width, height = subtracted_image.size

min_pixel = [255, 255, 255]
max_pixel = [0, 0, 0]

for x in range(width):
    for y in range(height):
        for i in range(3):
            min_pixel[i] = min(min_pixel[i], pixels[x, y][i])
            max_pixel[i] = max(max_pixel[i], pixels[x, y][i])

# Apply scaling to all pixels
for i in range(3):
    if max_pixel[i] != min_pixel[i]:
        for x in range(width):
            for y in range(height):
                temp = list(pixels[x, y])
                p = temp[i]
                p = pixels[x, y][i]
                
                temp[i]= int(255 * (p - min_pixel[i]) / (max_pixel[i] - min_pixel[i]))
                pixels[x, y] = tuple(temp)


subtracted_image.save("scaled_subtracted_image.png")
