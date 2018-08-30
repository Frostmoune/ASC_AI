import tensorflow as tf
import vis_lstm_model
import data_loader
import argparse
import numpy as np
import os

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--num_lstm_layers', type=int, default=3,
                       help='num_lstm_layers')
	parser.add_argument('--feature_length', type=int, default=4096,
                       help='feature_length')
	parser.add_argument('--rnn_size', type=int, default=512,
                       help='rnn_size')
	parser.add_argument('--embedding_size', type=int, default=512,
                       help='embedding_size'),
	parser.add_argument('--word_emb_dropout', type=float, default=0.75,
                       help='word_emb_dropout')
	parser.add_argument('--image_dropout', type=float, default=0.5,
                       help='image_dropout')
	parser.add_argument('--data_dir', type=str, default='Data',
                       help='Data directory')
	parser.add_argument('--image_features', type = str, default = 'vgg16',
                       help = 'Image features type')
	parser.add_argument('--batch_size', type=int, default=100,
                       help='Batch Size')
	parser.add_argument('--learning_rate', type=float, default=0.0015,
                       help='Batch Size')
	parser.add_argument('--epochs', type=int, default=41,
                       help='Expochs')
	parser.add_argument('--debug', type=bool, default=False,
                       help='Debug')
	parser.add_argument('--model_path', type=str, default = 'Data/Models/model40.ckpt',
                       help='Model Path')
	parser.add_argument('--version', type=int, default=1,
                       help='VQA data version')

	args = parser.parse_args()
	print("Reading QA DATA")
	# qa_data = data_loader.load_questions_answers(args)
	qa_data = data_loader.load_questions_answers(args.version, args.data_dir)

	print("Reading features")
	features, image_id_list = data_loader.load_features(args.data_dir, 'val', args.image_features)
	print("Features", features.shape)
	print("Image_id_list", image_id_list.shape)

	image_id_map = {}
	for i in range(len(image_id_list)):
		image_id_map[ image_id_list[i] ] = i

	ans_map = { qa_data['answer_vocab'][ans] : ans for ans in qa_data['answer_vocab']}

	model_options = {
		'num_lstm_layers' : args.num_lstm_layers,
		'rnn_size' : args.rnn_size,
		'embedding_size' : args.embedding_size,
		'word_emb_dropout' : args.word_emb_dropout,
		'image_dropout' : args.image_dropout,
		'feature_length' : args.feature_length,
		'lstm_steps' : qa_data['max_question_length'] + 1,
		'q_vocab_size' : len(qa_data['question_vocab']),
		'ans_vocab_size' : len(qa_data['answer_vocab'])
	}
	
	model = vis_lstm_model.Vis_lstm_model(model_options)
	input_tensors, t_prediction, t_ans_probab = model.build_generator()
	sess = tf.InteractiveSession()
	saver = tf.train.Saver()

	avg_accuracy = 0.0
	total = 0
	saver.restore(sess, args.model_path)
	
	batch_no = 0
	while (batch_no*args.batch_size) < len(qa_data['validation']):
		sentence, answer, batch_features = get_batch(batch_no, args.batch_size, 
			features, image_id_map, qa_data, 'val', args.image_features)
		
		pred, ans_prob = sess.run([t_prediction, t_ans_probab], feed_dict={
            input_tensors['features']:batch_features,
            input_tensors['sentence']:sentence,
        })
		
		batch_no += 1
		if args.debug:
			for idx, p in enumerate(pred):
				print(ans_map[p], ans_map[ np.argmax(answer[idx])])
		
		correct_predictions = np.equal(pred, np.argmax(answer, 1))
		correct_predictions = correct_predictions.astype('float32')
		accuracy = correct_predictions.mean()
		print("%d Acc: %f"%(batch_no, accuracy))
		avg_accuracy += accuracy
		total += 1
	
	print("Total Acc", avg_accuracy/total)


def get_batch(batch_no, batch_size, all_features, image_id_map, qa_data, split, image_feature):
	qa = None
	if split == 'train':
		qa = qa_data['training']
	else:
		qa = qa_data['validation']

	si = (batch_no * batch_size)%len(qa)
	ei = min(len(qa), si + batch_size)
	n = ei - si
	sentence = np.ndarray( (n, qa_data['max_question_length']), dtype = 'int32')
	answer = np.zeros( (n, len(qa_data['answer_vocab'])))
	if image_feature == 'vgg16':
		features = np.ndarray( (n, 4096) )
	elif image_feature == 'resnet101':
		features = np.ndarray( (n, 2048) )
	elif image_feature == 'inception_v3':
		features = np.ndarray( (n, 2048) )

	count = 0

	for i in range(si, ei):
		sentence[count,:] = qa[i]['question'][:]
		answer[count, qa[i]['answer']] = 1.0
		features_index = image_id_map[ qa[i]['image_id'] ]
		features[count,:] = all_features[features_index][:]
		count += 1
	
	return sentence, answer, features

if __name__ == '__main__':
	main()