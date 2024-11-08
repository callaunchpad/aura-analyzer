import argparse
import math
from typing import Tuple, Union

import cv2
import numpy as np
from PIL import Image
import imutils
import matplotlib as plt
from mediapipe.framework.formats import landmark_pb2
from mediapipe import solutions

MARGIN = 10  # pixels
ROW_SIZE = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
TEXT_COLOR = (255, 0, 0)  # red

# trying soemthing new 
# just want to test
# please delete later

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

def draw_landmarks_on_image(detection_result, img):
  face_landmarks_list = detection_result.face_landmarks

  # Loop through the detected faces to visualize.
  for idx in range(len(face_landmarks_list)):
    face_landmarks = face_landmarks_list[idx]

    # Draw the face landmarks.
    face_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
    face_landmarks_proto.landmark.extend([
      landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in face_landmarks
    ])

    solutions.drawing_utils.draw_landmarks(
        image=img,
        landmark_list=face_landmarks_proto,
        connections=mp.solutions.face_mesh.FACEMESH_TESSELATION,
        landmark_drawing_spec=None,
        connection_drawing_spec=mp.solutions.drawing_styles
        .get_default_face_mesh_tesselation_style())
    solutions.drawing_utils.draw_landmarks(
        image=img,
        landmark_list=face_landmarks_proto,
        connections=mp.solutions.face_mesh.FACEMESH_CONTOURS,
        landmark_drawing_spec=None,
        connection_drawing_spec=mp.solutions.drawing_styles
        .get_default_face_mesh_contours_style())
    solutions.drawing_utils.draw_landmarks(
        image=img,
        landmark_list=face_landmarks_proto,
        connections=mp.solutions.face_mesh.FACEMESH_IRISES,
          landmark_drawing_spec=None,
          connection_drawing_spec=mp.solutions.drawing_styles
          .get_default_face_mesh_iris_connections_style())

  return img

def plot_face_blendshapes_bar_graph(face_blendshapes):
  # Extract the face blendshapes category names and scores.
  face_blendshapes_names = [face_blendshapes_category.category_name for face_blendshapes_category in face_blendshapes]
  face_blendshapes_scores = [face_blendshapes_category.score for face_blendshapes_category in face_blendshapes]
  # The blendshapes are ordered in decreasing score value.
  face_blendshapes_ranks = range(len(face_blendshapes_names))

  fig, ax = plt.subplots(figsize=(12, 12))
  bar = ax.barh(face_blendshapes_ranks, face_blendshapes_scores, label=[str(x) for x in face_blendshapes_ranks])
  ax.set_yticks(face_blendshapes_ranks, face_blendshapes_names)
  ax.invert_yaxis()

  # Label each bar with values
  for score, patch in zip(face_blendshapes_scores, bar.patches):
    plt.text(patch.get_x() + patch.get_width(), patch.get_y(), f"{score:.4f}", va="top")

  ax.set_xlabel('Score')
  ax.set_title("Face Blendshapes")
  plt.tight_layout()
  plt.show()



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


# STEP 1: Import the necessary modules.
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision 

# STEP 2: Create an FaceDetector object.
# i want to add more code

base_options = python.BaseOptions(model_asset_path="../../face-detect/facedec/detector.tflite")
options = vision.FaceDetectorOptions(base_options=base_options)
detector = vision.FaceDetector.create_from_options(options)

# Creating Face-Landmarker Object
base_options_ld = python.BaseOptions(model_asset_path='../../face-detect/facedec/face_landmarker_v2_with_blendshapes.task')
options_ld = vision.FaceLandmarkerOptions(base_options=base_options_ld,
                                       output_face_blendshapes=True,
                                       output_facial_transformation_matrixes=True,
                                       num_faces=1)
detector_ld = vision.FaceLandmarker.create_from_options(options_ld)





parser = argparse.ArgumentParser(description="Changing WB of an input image.")
parser.add_argument("--input", "-i", help="Input image filename", dest="input", default="../example_images/00.JPG")
args = parser.parse_args()

# STEP 3: Load the input image.
IMAGE_FILE = args.input
image = mp.Image.create_from_file(IMAGE_FILE)

# STEP 4: Detect faces in the input image.
detection_result = detector.detect(image)
img_copy = np.copy(image.numpy_view())

img = cv2.imread(IMAGE_FILE)

detection_result_ld = detector_ld.detect(image)

annotated_ld_image = draw_landmarks_on_image(detection_result_ld,img)

# STEP 5: Process the detection result. In this case, visualize it.
image_copy = np.copy(image.numpy_view())
annotated_image, cropped = visualize(image_copy, detection_result)
cropped = imutils.resize(cropped, width=800)
rgb_annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
rgb_annotated_image = imutils.resize(rgb_annotated_image, width=800)

# annotated_image_2 = draw_landmarks_on_image(image.numpy_view(), annotated_image_2)
# SAVE CROPPED AS IMAGE: cv2.imwrite("cropped.jpg", cropped)
cv2.imwrite("../output-imgs/redbox.jpg", rgb_annotated_image)
cv2.imwrite("../output-imgs/cropped.jpg", cropped)
cv2.imwrite("../output-imgs/facelandmark.jpg", annotated_ld_image)


cv2.imshow("Gotchaface", rgb_annotated_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imshow("CROPPED", cropped)
cv2.imshow("Annotated Image", cv2.cvtColor(annotated_ld_image, cv2.COLOR_RGB2BGR))
cv2.waitKey(0)
cv2.destroyAllWindows()
