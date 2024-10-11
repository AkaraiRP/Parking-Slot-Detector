from image_processing import crop_and_transform

"""
sorry for the jank, i forgot the correct way to do python tests xddd but this should do for now
"""

def test1():
    """test for crop_and_transform: should crop out green car in 5.png"""
    image_path = 'sample_images\\5.png'
    points = [(433, 22), (479, 19), (491, 76), (439, 76)]

    cropped_image = crop_and_transform(image_path, points)
    cropped_image.show()  # Displays the image with default application

if __name__ == "__main__":
    test1()