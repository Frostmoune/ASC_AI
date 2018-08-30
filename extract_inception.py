import tensorflow as tf
from scipy import misc
from os import listdir
from os.path import isfile, join
import data_loader
import utils
import argparse
import numpy as np
import pickle
import h5py
import time
import tensorflow.contrib.slim.nets as nets

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', type = str, default = 'train',
                        help = 'train/val/test')
    parser.add_argument('--model_path', type = str, default = './Data/Inception/inception_v3.ckpt',
                        help = 'Pretrained RESNET Model')
    parser.add_argument('--data_dir', type = str, default = 'Data',
                        help = 'Data directory')
    parser.add_argument('--batch_size', type = int, default = 10,
                        help = 'Batch Size')

    args = parser.parse_args()

    slim = tf.contrib.slim
    inception = nets.inception

    config = tf.ConfigProto(allow_soft_placement = True)
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.9
    sess = tf.InteractiveSession(config = config)
    sess.run(tf.global_variables_initializer())

    if args.split == 'test':
        all_data = data_loader.load_test_questions()
        qa_data = all_data['testing']
    else:
        all_data = data_loader.load_questions_answers(args)
        if args.split == "train":
            qa_data = all_data['training']
        else:
            qa_data = all_data['validation']

    image_ids = {}
    for qa in qa_data:
        image_ids[qa['image_id']] = 1

    image_id_list = [img_id for img_id in image_ids]
    print("Total Images", len(image_id_list))

    length = 100 if args.split == 'test' else len(image_id_list)
    ince_pool = np.ndarray( (length, 2048 ) )
    idx = 0
    SIZE = 299
    flag = 0


    while idx < length:
        if idx % 500 == 0:
            flag = 0
            tf.reset_default_graph()
        with tf.Graph().as_default():
            with tf.Session() as sess:
                while idx < length:
                    start = time.clock()
                    image_batch = np.ndarray( (args.batch_size, SIZE, SIZE, 3), dtype = np.float32)

                    count = 0
                    for i in range(0, args.batch_size):
                        if idx >= len(image_id_list):
                            break
                        if args.split == 'test':
                            image_file = join(args.data_dir, '%s2015/COCO_%s2015_%.12d.jpg'%(args.split, args.split, image_id_list[idx]) )
                        else:
                            image_file = join(args.data_dir, '%s2014/COCO_%s2014_%.12d.jpg'%(args.split, args.split, image_id_list[idx]) )
                        image_batch[i,:,:,:] = utils.load_image_array(image_file, size = SIZE)
                        idx += 1
                        count += 1
                    
                    with slim.arg_scope(inception.inception_v3_arg_scope()):
                        logits, end_points = inception.inception_v3(image_batch[0:count, :, :, :], num_classes = 1001, dropout_keep_prob = 1.0, is_training = False, reuse = tf.AUTO_REUSE)
                        if not flag:
                            vals = slim.get_model_variables()
                            init_fn = slim.assign_from_checkpoint_fn(args.model_path, vals)
                            init_fn(sess)
                            flag = 1

                    ince_pool_batch = sess.run([end_points['PreLogits']])
                    if idx % 10 == 0:
                        ince_pool_batch = ince_pool_batch[0].reshape((args.batch_size, 2048))
                    else:
                        ince_pool_batch = ince_pool_batch[0].reshape((idx % args.batch_size, 2048))
                    ince_pool[(idx - count):idx, :] = ince_pool_batch[0:count, :]
                    end = time.clock()
                    print("Time for batch 10 photos", end - start)
                    print("Hours For Whole Dataset" , (len(image_id_list) * 1.0)*(end - start)/60.0/60.0/10.0)

                    print("Images Processed", idx)
                    if idx % 500 == 0:
                        break

    print("Saving inception features")
    h5f_res5c = h5py.File( join(args.data_dir, args.split + '_inception.h5'), 'w')
    h5f_res5c.create_dataset('inception_features', data = ince_pool)
    h5f_res5c.close()

    print("Saving image id list")
    h5f_image_id_list = h5py.File( join(args.data_dir, args.split + '_image_id_list.h5'), 'w')
    h5f_image_id_list.create_dataset('image_id_list', data = image_id_list)
    h5f_image_id_list.close()
    print("Done!")

if __name__ == '__main__':
	main()