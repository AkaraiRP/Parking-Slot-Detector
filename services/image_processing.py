import cv2
import numpy as np
from PIL import Image

def crop_and_transform(input_image, points: list[tuple[int]]):
    """
    Crops out a quadrilateral-shaped region from an image based on four pixel coordinates.

    Args:
    - image_path: Path to the input image (string) or PIL Image object.
    - points: A list of four tuples, each representing (x, y) coordinates of the quadrilateral's corners.
              Example: [(x1, y1), (x2, y2), (x3, y3), (x4, y4)].

    Returns:
    - Cropped sub-image (PIL Image).
    """

    # TODO: troubleshoot why ifinstance doesn't like Image as the second arg
    # if isinstance(input_image, Image):
    #     image = np.array(input_image)
    
    if isinstance(input_image, str):
        # Load image using OpenCV
        image = cv2.imread(input_image)
        if image is None:
            raise ValueError("Image not found or invalid image path.")
    else:
        # TODO: remove try/except block and image = np.array(input_image) if isinstance problem is fixed
        try:
            image = np.array(input_image) 
        except:
            raise TypeError("Image must be a valid path (string) or PIL Image")

    # Convert the points to NumPy array
    if len(points) != 4:
        raise ValueError("Please provide 4 points")
    elif [len(point) for point in points] != [2, 2, 2, 2]:
        print(f"len(point)s {(len(point) for point in points)}")
        raise ValueError("Points must be provided as (x, y) coordinates")
    
    points = np.array(points, dtype="float32")

    # Determine the width and height of the cropped image
    width_A = np.linalg.norm(points[0] - points[1])
    width_B = np.linalg.norm(points[2] - points[3])
    max_width = int(max(width_A, width_B))

    height_A = np.linalg.norm(points[1] - points[2])
    height_B = np.linalg.norm(points[3] - points[0])
    max_height = int(max(height_A, height_B))

    # Define the destination points for the transformed image
    dest_points = np.array([[0, 0], [max_width - 1, 0], [max_width - 1, max_height - 1], [0, max_height - 1]], dtype="float32")

    # Compute the perspective transformation matrix
    matrix = cv2.getPerspectiveTransform(points, dest_points)

    # Perform the warp perspective to obtain the warped (rectangular) image
    warped_image = cv2.warpPerspective(image, matrix, (max_width, max_height))

    # Convert the result to a PIL image and return it
    cropped_image = Image.fromarray(cv2.cvtColor(warped_image, cv2.COLOR_BGR2RGB))
    print(f"type: {type(cropped_image)}")

    return cropped_image
