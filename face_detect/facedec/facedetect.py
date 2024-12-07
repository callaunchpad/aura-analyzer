import argparse
import math
import os
from subprocess import STDOUT, check_call
from typing import Tuple, Union

check_call(["apt-get", "update"], stdout=open(os.devnull, "wb"), stderr=STDOUT)
check_call(["apt-get", "install", "-y", "libgl1"], stdout=open(os.devnull, "wb"), stderr=STDOUT)
check_call(["apt-get", "install", "-y", "libglib2.0-0"], stdout=open(os.devnull, "wb"), stderr=STDOUT)
check_call(["apt-get", "update"], stdout=open(os.devnull, "wb"), stderr=STDOUT)

import cv2
import imutils
import numpy as np
from PIL import Image

MARGIN = 10  # pixels
ROW_SIZE = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
TEXT_COLOR = (255, 0, 0)  # red


def _normalized_to_pixel_coordinates(
    normalized_x: float, normalized_y: float, image_width: int, image_height: int
) -> Union[None, Tuple[int, int]]:
    """Converts normalized value pair to pixel coordinates."""

    # Checks if the float value is between 0 and 1.
    def is_valid_normalized_value(value: float) -> bool:
        return (value > 0 or math.isclose(0, value)) and (value < 1 or math.isclose(1, value))

    if not (is_valid_normalized_value(normalized_x) and is_valid_normalized_value(normalized_y)):
        # TODO: Draw coordinates even if it's outside of the image bounds.
        return None
    x_px = min(math.floor(normalized_x * image_width), image_width - 1)
    y_px = min(math.floor(normalized_y * image_height), image_height - 1)
    return x_px, y_px


def visualize(image, detection_result) -> np.ndarray:
    """Draws bounding boxes and keypoints on the input image and return it.
    Args:
      image: The input RGB image.
      detection_result: The list of all "Detection" entities to be visualize.
    Returns:
      Image with bounding boxes.
    """
    annotated_image = image.copy()
    height, width, _ = image.shape

    for detection in detection_result.detections:
        # Draw bounding_box
        bbox = detection.bounding_box
        start_point = (int)(bbox.origin_x * 0.95), (int)(bbox.origin_y * 0.75)
        end_point = bbox.origin_x + bbox.width + (int)(bbox.origin_x * 0.05), bbox.origin_y + bbox.height
        cv2.rectangle(annotated_image, start_point, end_point, TEXT_COLOR, 3)
        # blue cropped = image[start_point[1] : end_point[1], start_point[0] : end_point[0]]
        cropped = cv2.cvtColor(image[start_point[1] : end_point[1], start_point[0] : end_point[0]], cv2.COLOR_BGR2RGB)
        # WATERMELON cropped = cv2.cvtColor(image[start_point[1]:end_point[1], start_point[0]:end_point[0]], cv2.COLOR_BGR2HSV)
        # Draw keypoints
        for keypoint in detection.keypoints:
            keypoint_px = _normalized_to_pixel_coordinates(keypoint.x, keypoint.y, width, height)
            color, thickness, radius = (0, 255, 0), 2, 2
            cv2.circle(annotated_image, keypoint_px, thickness, color, radius)

        # Draw label and score
        category = detection.categories[0]
        category_name = category.category_name
        category_name = "" if category_name is None else category_name
        probability = round(category.score, 2)
        result_text = category_name + " (" + str(probability) + ")"
        text_location = (MARGIN + bbox.origin_x, MARGIN + ROW_SIZE + bbox.origin_y)
        cv2.putText(
            annotated_image, result_text, text_location, cv2.FONT_HERSHEY_PLAIN, FONT_SIZE, TEXT_COLOR, FONT_THICKNESS
        )

    return annotated_image, cropped


if __name__ == "__main__":

    # STEP 1: Import the necessary modules.
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision

    # STEP 2: Create an FaceDetector object.
    base_options = python.BaseOptions(model_asset_path="face_detect/facedec/detector.tflite")
    options = vision.FaceDetectorOptions(base_options=base_options)
    detector = vision.FaceDetector.create_from_options(options)

    parser = argparse.ArgumentParser(description="Changing WB of an input image.")
    parser.add_argument("--input", "-i", help="Input image filename", dest="input", default="../example_images/00.JPG")
    parser.add_argument("--file_name", dest="file_name")
    args = parser.parse_args()

    # STEP 3: Load the input image.
    IMAGE_FILE = args.input
    image = mp.Image.create_from_file(IMAGE_FILE)

    # STEP 4: Detect faces in the input image.
    detection_result = detector.detect(image)

    # STEP 5: Process the detection result. In this case, visualize it.
    image_copy = np.copy(image.numpy_view())
    annotated_image, cropped = visualize(image_copy, detection_result)
    cropped = imutils.resize(cropped, width=800)
    cropped_pil = Image.fromarray(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))
    img = cv2.imread(IMAGE_FILE)
    rgb_annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
    rgb_annotated_image = imutils.resize(rgb_annotated_image, width=800)
    rgb_annotated_image_pil = Image.fromarray(cv2.cvtColor(rgb_annotated_image, cv2.COLOR_BGR2RGB))

    out_dir = "combined_demo/output-imgs/"
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    rgb_annotated_image_pil.save(os.path.join(out_dir, f"{args.file_name}-redbox.jpg"))
    cropped_pil.save(os.path.join(out_dir, f"{args.file_name}-cropped.jpg"))

    # cv2.imwrite("combined_demo/output-imgs/redbox.jpg", rgb_annotated_image)
    # cv2.imwrite("combined_demo/output-imgs/cropped.jpg", cropped)

    # cv2.imshow("Gotchaface", rgb_annotated_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # cv2.imshow("CROPPED", cropped)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
