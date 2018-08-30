import numpy as np
from scipy import misc
import tensorflow as tf
import h5py
from os.path import isfile, join

# VGG 16 accepts RGB channel 0 to 1 (This tensorflow model).
def load_image_array(image_file, size = 224):
	img = misc.imread(image_file)
	# GRAYSCALE
	if len(img.shape) == 2:
		img_new = np.ndarray( (img.shape[0], img.shape[1], 3), dtype = 'float32')
		img_new[:,:,0] = img
		img_new[:,:,1] = img
		img_new[:,:,2] = img
		img = img_new

	img_resized = misc.imresize(img, (size, size))
	return (img_resized/255.0).astype('float32')

def extract_test_fc7_features(test_data, model_path):
	if isfile('Data/test_fc7_features.h5'):
		with h5py.File('Data/test_fc7_features.h5', 'r') as hf:
			print("Loading test_fc7_features...")
			fc7_features = np.array(hf.get('test_fc7_features'))
			if isfile('Data/test_image_list.h5'):
				with h5py.File('Data/test_image_list.h5', 'r') as hf2:
					print("Loading test_image_list...")
					image_to_id = np.array(hf2.get('test_image_list'))
					return fc7_features, image_to_id

	vgg_file = open(model_path, 'rb')
	vgg16raw = vgg_file.read()
	vgg_file.close()

	graph_def = tf.GraphDef()
	graph_def.ParseFromString(vgg16raw)
	images = tf.placeholder("float32", [None, 224, 224, 3])
	tf.import_graph_def(graph_def, input_map={ "images": images })
	graph = tf.get_default_graph()
	fc7_tensor = graph.get_tensor_by_name("import/Relu_1:0")
	sess = tf.Session()
	

	fc7_features = np.zeros((81435, 4096))
	image_to_id = [0 for i in range(len(test_data))]
	id_is_visit = {}
	image_list_id = 0

	for i, now_image in enumerate(test_data):
		if i > 100:
			break
		if now_image['image_id'] in id_is_visit:
			image_to_id[i] = id_is_visit[now_image['image_id']]
			continue

		now_image_path = 'Data/test2015/COCO_test2015_%.12d.jpg'%(now_image['image_id'])
		image_array = load_image_array(now_image_path)
		image_feed = np.ndarray((1, 224, 224, 3))
		image_feed[0:,:,:] = image_array
		feed_dict  = { images : image_feed }
		now_fc7_features = sess.run(fc7_tensor, feed_dict = feed_dict)
		fc7_features[image_list_id, :] = now_fc7_features

		image_to_id[i] = image_list_id
		id_is_visit[now_image['image_id']] = image_list_id
		image_list_id += 1
		print("Image %d finished."%(image_list_id + 1))

	
	sess.close()
	print("Saving fc7 features")
	h5f_fc7 = h5py.File('Data/test_fc7_features.h5', 'w')
	h5f_fc7.create_dataset('test_fc7_features', data = fc7_features)
	h5f_fc7.close()

	print("Saving image list")
	h5f_fc7 = h5py.File('Data/test_image_list.h5', 'w')
	h5f_fc7.create_dataset('test_image_list', data = image_to_id)
	h5f_fc7.close()
	
	return fc7_features, image_to_id

# FOR PREDICTION ON A SINGLE IMAGE
def extract_fc7_features(image_path, model_path):
	vgg_file = open(model_path, 'rb')
	vgg16raw = vgg_file.read()
	vgg_file.close()

	graph_def = tf.GraphDef()
	graph_def.ParseFromString(vgg16raw)
	images = tf.placeholder("float32", [None, 224, 224, 3])
	tf.import_graph_def(graph_def, input_map={ "images": images })
	graph = tf.get_default_graph()

	sess = tf.Session()
	image_array = load_image_array(image_path)
	image_feed = np.ndarray((1, 224, 224, 3))
	image_feed[0:,:,:] = image_array
	feed_dict  = { images : image_feed }
	fc7_tensor = graph.get_tensor_by_name("import/Relu_1:0")
	fc7_features = sess.run(fc7_tensor, feed_dict = feed_dict)
	sess.close()
	return fc7_features