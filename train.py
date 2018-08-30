import tensorflow as tf
import vis_lstm_model
import data_loader
import argparse
import numpy as np
import os
import time

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--num_lstm_layers', type = int, default = 2,
                       help = 'num_lstm_layers')
	parser.add_argument('--feature_length', type = int, default = 4096,
                       help = 'feature_length')
	parser.add_argument('--rnn_size', type = int, default = 512,
                       help = 'rnn_size')
	parser.add_argument('--embedding_size', type = int, default = 512,
                       help = 'embedding_size'),
	parser.add_argument('--word_emb_dropout', type = float, default = 0.75,
                       help = 'word_emb_dropout')
	parser.add_argument('--image_dropout', type = float, default = 0.5,
                       help = 'image_dropout')
	parser.add_argument('--data_dir', type = str, default = 'Data',
                       help = 'Data directory')
	parser.add_argument('--image_features', type = str, default = 'vgg16',
                       help = 'Image features type')
	parser.add_argument('--batch_size', type = int, default = 100,
                       help = 'Batch Size')
	parser.add_argument('--learning_rate', type = float, default = 0.001,
                       help = 'Batch Size')
	parser.add_argument('--epochs', type = int, default = 20,
                       help = 'Expochs')
	parser.add_argument('--debug', type = bool, default = False,
                       help = 'Debug')
	parser.add_argument('--resume_model', type = str, default = None,
                       help = 'Trained Model Path')
	parser.add_argument('--version', type = int, default = 1,
                       help = 'VQA data version')

	args = parser.parse_args()
	print("Reading QA DATA")
	qa_data = data_loader.load_questions_answers(args.version, args.data_dir)
	print(len(qa_data['question_vocab']))
	
	print("Reading features")
	features, image_id_list = data_loader.load_features(args.data_dir, 'train', args.image_features)

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
	input_tensors, t_loss, t_accuracy, t_p = model.build_model()
	train_op = tf.train.AdamOptimizer(args.learning_rate).minimize(t_loss)
	sess = tf.InteractiveSession()
	sess.run(tf.global_variables_initializer())

	saver = tf.train.Saver()
	if args.resume_model:
		saver.restore(sess, args.resume_model)

	for i in range(args.epochs):
		batch_no = 0
		if not os.path.exists('Data/Logs/{}'.format(i)):
			os.mkdir('Data/Logs/{}'.format(i))
		merged_summary = tf.summary.merge_all()
		summary_writer = tf.summary.FileWriter('Data/Logs/{}'.format(i))
		summary_writer.add_graph(sess.graph)

		start = time.clock()
		while (batch_no*args.batch_size) < len(qa_data['training']):
			sentence, answer, batch_features = get_training_batch(batch_no, args.batch_size, features, image_id_map, qa_data, 'train', args.image_features)
			now_summary, _, loss_value, accuracy, pred = sess.run([merged_summary, train_op, t_loss, t_accuracy, t_p],
				feed_dict = {
					input_tensors['features']:batch_features,
					input_tensors['sentence']:sentence,
					input_tensors['answer']:answer
				}
			)
			summary_writer.add_summary(now_summary, batch_no) 
			batch_no += 1
			if args.debug:
				for idx, p in enumerate(pred):
					print(ans_map[p], ans_map[ np.argmax(answer[idx])])

				print("Loss", loss_value, batch_no, i)
				print("Accuracy", accuracy)
				print("---------------")
			else:
				print("Loss", loss_value, batch_no, i)
				print("Training Accuracy", accuracy)
		end = time.clock()
		print("循环，时间：", i, end - start)
		# if i % 5 == 0 and not os.path.exists("Data/Models/" + str(i // 5)):
		# 	os.mkdir("Data/Models/" + str(i // 5))

		save_path = saver.save(sess, "Data/Models/model{}.ckpt".format(i))

def get_training_batch(batch_no, batch_size, all_features, image_id_map, qa_data, split, image_feature):
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
		feature = np.ndarray( (n, 4096) )
	elif image_feature == 'resnet101':
		feature = np.ndarray( (n, 2048) )
	elif image_feature == 'inception_v3':
		feature = np.ndarray( (n, 2048) )

	count = 0
	for i in range(si, ei):
		sentence[count,:] = qa[i]['question'][:]
		answer[count, qa[i]['answer']] = 1.0
		now_index = image_id_map[ qa[i]['image_id'] ]
		feature[count,:] = all_features[now_index][:]
		count += 1
	
	return sentence, answer, feature

if __name__ == '__main__':
	main()