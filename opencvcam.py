from basecamera import BaseCamera
import cv2
import tensorflow as tf
import os
from object_detection.utils import visualization_utils, label_map_util, ops
import numpy as np
import face_recognizer as fr

# class ObjectDetector(object):
#     def __init__(self,model_name):
#         self.model_name = model_name
#         self.graph = tf.Graph()
#         self.num_class = 1
#         self.initialize_graph()
#         self.initialize_labels()
#         self.session = None
    
#     def __del__(self):
#         if self.session is not None:
#             self.session.close()

#     def run_inference_for_single_image(self,image,session):
#         ops = tf.get_default_graph().get_operations()
#         all_tensor_names = {output.name for op in ops for output in op.outputs}
#         tensor_dict = {}
#         for key in [
#             'num_detections', 'detection_boxes', 'detection_scores',
#             'detection_classes', 'detection_masks'
#         ]:
#             tensor_name = key + ':0'
#             if tensor_name in all_tensor_names:
#                 tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
#                     tensor_name)
#         if 'detection_masks' in tensor_dict:
#             # The following processing is only for single image
#             detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
#             detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
#             # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
#             real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
#             detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
#             detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
#             detection_masks_reframed = ops.reframe_box_masks_to_image_masks(
#                 detection_masks, detection_boxes, image.shape[0], image.shape[1])
#             detection_masks_reframed = tf.cast(
#                 tf.greater(detection_masks_reframed, 0.5), tf.uint8)
#             # Follow the convention by adding back the batch dimension
#             tensor_dict['detection_masks'] = tf.expand_dims(
#                 detection_masks_reframed, 0)
#         image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

#         # Run inference
#         output_dict = session.run(tensor_dict,
#                                 feed_dict={image_tensor: np.expand_dims(image, 0)})

#         # all outputs are float32 numpy arrays, so convert types as appropriate
#         output_dict['num_detections'] = int(output_dict['num_detections'][0])
#         output_dict['detection_classes'] = output_dict[
#             'detection_classes'][0].astype(np.uint8)
#         output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
#         output_dict['detection_scores'] = output_dict['detection_scores'][0]
#         if 'detection_masks' in output_dict:
#             output_dict['detection_masks'] = output_dict['detection_masks'][0]
#         visualization_utils.visualize_boxes_and_labels_on_image_array(
#             image,
#             output_dict['detection_boxes'],
#             output_dict['detection_classes'],
#             output_dict['detection_scores'],
#             self.category_index,
#             use_normalized_coordinates=True,
#             line_thickness=8)
#         return output_dict, image

#     def initialize_graph(self):
#         model_path = os.path.join(self.model_name,'frozen_inference_graph.pb')
#         with self.graph.as_default():
#             temp_graph_def = tf.GraphDef()
#             with tf.gfile.GFile(model_path,'rb') as f:
#                 ser_graph = f.read()
#                 temp_graph_def.ParseFromString(ser_graph)
#                 tf.import_graph_def(temp_graph_def,name='')
    
#     def initialize_labels(self):
#         path_to_label = os.path.join(self.model_name,'label.pbtxt')        
#         label_map = label_map_util.load_labelmap(path=path_to_label)
#         categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=self.num_class, use_display_name=True)
#         self.category_index = label_map_util.create_category_index(categories)

    
# detector = ObjectDetector("facedetector")

class Camera(BaseCamera):
    def __init__(self):
        # self.detector = ObjectDetector("facedetector")
        return super().__init__()
    @staticmethod
    def frames():
        cap = cv2.VideoCapture(0)
        facer = fr.face_recognizer(face_dir="face/")
        if not cap.isOpened():
            raise RuntimeError('Camera not found')
        while True:
            _, img = cap.read()
            face_locs, face_names = facer.runinference(img, tolerance=0.6, prescale=0.25, upsample=2)
            img = facer.display(img, face_locs, face_names, 0.25)
            yield cv2.imencode('.jpg',img)[1].tobytes()