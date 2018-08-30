import json
import argparse
from os.path import isfile, join
import re
import numpy as np
import pprint
import pickle
import nltk

WNL = nltk.stem.WordNetLemmatizer()

def prepare_training_data(version = 1, data_dir = 'Data'):
	if version == 1:
		t_q_json_file = join(data_dir, 'MultipleChoice_mscoco_train2014_questions.json')
		t_a_json_file = join(data_dir, 'mscoco_train2014_annotations.json')

		v_q_json_file = join(data_dir, 'MultipleChoice_mscoco_val2014_questions.json')
		v_a_json_file = join(data_dir, 'mscoco_val2014_annotations.json')

		qa_data_file = join(data_dir, 'qa_data_file1.pkl')
		vocab_file = join(data_dir, 'vocab_file1.pkl')
	else:
		t_q_json_file = join(data_dir, 'v2_OpenEnded_mscoco_train2014_questions.json')
		t_a_json_file = join(data_dir, 'v2_mscoco_train2014_annotations.json')

		v_q_json_file = join(data_dir, 'v2_OpenEnded_mscoco_val2014_questions.json')
		v_a_json_file = join(data_dir, 'v2_mscoco_val2014_annotations.json')
		qa_data_file = join(data_dir, 'qa_data_file2.pkl')
		vocab_file = join(data_dir, 'vocab_file2.pkl')

	# IF ALREADY EXTRACTED
	# qa_data_file = join(data_dir, 'qa_data_file{}.pkl'.format(version))
	# if isfile(qa_data_file):
	# 	with open(qa_data_file) as f:
	# 		data = pickle.load(f)
	# 		return data

	print("Loading Training questions")
	with open(t_q_json_file) as f:
		t_questions = json.loads(f.read())
	
	print("Loading Training anwers")
	with open(t_a_json_file) as f:
		t_answers = json.loads(f.read())

	print("Loading Val questions")
	with open(v_q_json_file) as f:
		v_questions = json.loads(f.read())
	
	print("Loading Val answers")
	with open(v_a_json_file) as f:
		v_answers = json.loads(f.read())
	
	print("Ans", len(t_answers['annotations']), len(v_answers['annotations']))
	print("Qu", len(t_questions['questions']), len(v_questions['questions']))

	answers = t_answers['annotations'] + v_answers['annotations']
	questions = t_questions['questions'] + v_questions['questions']
	
	stop_vocab = ['a', 'an', 'the'] 
	answer_vocab = make_answer_vocab(answers)
	question_vocab, max_question_length = make_questions_vocab(questions, answers, answer_vocab, stop_vocab)
	print("Max Question Length", max_question_length)

	word_regex = re.compile(r'\w+')
	training_data = []
	for i,question in enumerate( t_questions['questions']):
		ans = vocab_handle(t_answers['annotations'][i]['multiple_choice_answer'])
		if ans in answer_vocab:
			training_data.append({
				'image_id' : t_answers['annotations'][i]['image_id'],
				'question' : np.zeros(max_question_length),
				'answer' : answer_vocab[ans]
				})
			question_words = re.findall(word_regex, question['question'])
			question_words = list(filter(lambda x: vocab_handle(x) not in stop_vocab, question_words))

			base = max_question_length - len(question_words)
			for j in range(0, len(question_words)):
				try:
					training_data[-1]['question'][base + j] = question_vocab[ vocab_handle(question_words[j]) ]
				except Exception as e:
					print(question_words)
					print(question_words[j])
					print(vocab_handle(question_words[j]))
					input()
	print("Training Data", len(training_data))

	val_data = []
	for i,question in enumerate( v_questions['questions']):
		ans = vocab_handle(v_answers['annotations'][i]['multiple_choice_answer'])
		if ans in answer_vocab:
			val_data.append({
				'image_id' : v_answers['annotations'][i]['image_id'],
				'question' : np.zeros(max_question_length),
				'answer' : answer_vocab[ans]
				})
			question_words = re.findall(word_regex, question['question'])
			question_words = list(filter(lambda x: vocab_handle(x) not in stop_vocab, question_words))

			base = max_question_length - len(question_words)
			for i in range(0, len(question_words)):
				val_data[-1]['question'][base + i] = question_vocab[ vocab_handle(question_words[i]) ]
	print("Validation Data", len(val_data))

	data = {
		'training' : training_data,
		'validation' : val_data,
		'answer_vocab' : answer_vocab,
		'question_vocab' : question_vocab,
		'max_question_length' : max_question_length
	}

	print("Saving qa_data")
	with open(qa_data_file, 'wb') as f:
		pickle.dump(data, f)

	with open(vocab_file, 'wb') as f:
		vocab_data = {
			'answer_vocab' : data['answer_vocab'],
			'question_vocab' : data['question_vocab'],
			'max_question_length' : data['max_question_length']
		}
		pickle.dump(vocab_data, f)

	return data

def vocab_handle(voca):
	now_voca = voca.lower()
	now_voca = WNL.lemmatize(now_voca)
	if now_voca == 'are' or now_voca == 'is' or now_voca == 'am':
		now_voca = 'be'
	return now_voca

def make_test_questions_vocab(questions, stop_vocab):
	word_regex = re.compile(r'\w+')
	question_frequency = {}

	max_question_length = 0
	for i, question in enumerate(questions):
		count = 0
		question_words = re.findall(word_regex, question['question'])
		for qw in question_words:
			qword = vocab_handle(qw)
			if qword in stop_vocab:
				continue
			if qword in question_frequency:
				question_frequency[qword] += 1
			else:
				question_frequency[qword] = 1
			count += 1
		if count > max_question_length:
			max_question_length = count

	qw_freq_threhold = 0
	qw_tuples = [ (-frequency, qw) for qw, frequency in question_frequency.items()]

	qw_vocab = {}
	for i, qw_freq in enumerate(qw_tuples):
		frequency = -qw_freq[0]
		qw = qw_freq[1]
		# print frequency, qw
		if frequency > qw_freq_threhold:
			# +1 for accounting the zero padding for batc training
			qw_vocab[qw] = i + 1
		else:
			break

	qw_vocab['UNK'] = len(qw_vocab) + 1

	return qw_vocab, max_question_length

def prepare_testing_data(version = 1, data_dir = 'Data'):
	test_q_json_file = join(data_dir, 'MultipleChoice_mscoco_test2015_questions.json')
	test_data_file = join(data_dir, 'test_data_file.pkl')
	test_voca_file = join(data_dir, 'test_voca_file.pkl')

	print("Loading Test questions")
	with open(test_q_json_file) as f:
		test_questions = json.loads(f.read())

	stop_vocab = ['a', 'an', 'the']
	question_vocab, max_question_length = make_test_questions_vocab(test_questions['questions'], stop_vocab)
	print("Max Question Length", max_question_length)

	word_regex = re.compile(r'\w+')
	testing_data = []
	for i, question in enumerate(test_questions['questions']):
		testing_data.append({
			'image_id' : test_questions['questions'][i]['image_id'],
			'question_features' : np.zeros(max_question_length),
			'question' : question['question'],
		})
		question_words = re.findall(word_regex, question['question'])
		question_words = list(filter(lambda x: vocab_handle(x) not in stop_vocab, question_words))

		base = max_question_length - len(question_words)
		for i in range(0, len(question_words)):
			testing_data[-1]['question_features'][base + i] = question_vocab[ vocab_handle(question_words[i]) ]
	print("Testing Data", len(testing_data))

	data = {
		'testing' : testing_data,
		'question_vocab' : question_vocab,
		'max_question_length' : max_question_length
	}

	with open(test_data_file, 'wb') as f:
		pickle.dump(data, f)

	with open(test_voca_file, 'wb') as f:
		vocab_data = {
			'question_vocab' : data['question_vocab'],
			'max_question_length' : data['max_question_length']
		}
		pickle.dump(vocab_data, f)

def load_test_questions(version = 1, data_dir = 'Data'):
	test_data_file = join(data_dir, 'test_data_file.pkl')
	if isfile(test_data_file):
		data = pickle.load(open(test_data_file, 'rb'), encoding = 'iso-8859-1')
		return data

def load_test_voca(version = 1, data_dir = 'Data'):
	test_voca_file = join(data_dir, 'test_voca_file.pkl')
	if isfile(test_voca_file):
		data = pickle.load(open(test_voca_file, 'rb'), encoding = 'iso-8859-1')
		return data
	
def load_questions_answers(version = 1, data_dir = 'Data'):
	qa_data_file = join(data_dir, 'qa_data_file1.pkl')
	
	if isfile(qa_data_file):
		data = pickle.load(open(qa_data_file, 'rb'), encoding = 'iso-8859-1')
		return data

def get_question_answer_vocab(version = 1, data_dir = 'Data'):
	vocab_file = join(data_dir, 'vocab_file1.pkl')
	vocab_data = pickle.load(open(vocab_file, 'rb'), encoding = 'iso-8859-1')
	return vocab_data

def make_answer_vocab(answers):
	top_n = 1000
	answer_frequency = {} 
	for annotation in answers:
		answer = vocab_handle(annotation['multiple_choice_answer'])
		if answer in answer_frequency:
			answer_frequency[answer] += 1
		else:
			answer_frequency[answer] = 1

	answer_frequency_tuples = [ (-frequency, answer) for answer, frequency in answer_frequency.items()]
	answer_frequency_tuples.sort()
	answer_frequency_tuples = answer_frequency_tuples[0:top_n-1]

	answer_vocab = {}
	for i, ans_freq in enumerate(answer_frequency_tuples):
		# print i, ans_freq
		ans = ans_freq[1]
		answer_vocab[ans] = i

	answer_vocab['UNK'] = top_n - 1
	return answer_vocab

def make_questions_vocab(questions, answers, answer_vocab, stop_vocab):
	word_regex = re.compile(r'\w+')
	question_frequency = {}

	max_question_length = 0
	for i,question in enumerate(questions):
		ans = vocab_handle(answers[i]['multiple_choice_answer'])
		count = 0
		if ans in answer_vocab:
			question_words = re.findall(word_regex, question['question'])
			for qw in question_words:
				qword = vocab_handle(qw)
				if qword in stop_vocab:
					continue
				if qword in question_frequency:
					question_frequency[qword] += 1
				else:
					question_frequency[qword] = 1
				count += 1
		if count > max_question_length:
			max_question_length = count

	qw_freq_threhold = 0
	qw_tuples = [ (-frequency, qw) for qw, frequency in question_frequency.items()]
	# qw_tuples.sort()

	qw_vocab = {}
	for i, qw_freq in enumerate(qw_tuples):
		frequency = -qw_freq[0]
		qw = qw_freq[1]
		# print frequency, qw
		if frequency > qw_freq_threhold:
			# +1 for accounting the zero padding for batc training
			qw_vocab[qw] = i + 1
		else:
			break

	qw_vocab['UNK'] = len(qw_vocab) + 1

	return qw_vocab, max_question_length


def load_features(data_dir, split, image_feature):
	import h5py
	features = None
	image_id_list = None
	if image_feature == 'vgg16':
		with h5py.File( join( data_dir, (split + '_fc7.h5')), 'r') as hf:
			features = np.array(hf.get('fc7_features'))
	elif image_feature == 'resnet101':
		with h5py.File( join( data_dir, (split + '_res5c.h5')), 'r') as hf:
			features = np.array(hf.get('res5c_features'))
	elif image_feature == 'inception_v3':
		with h5py.File( join( data_dir, (split + '_inception.h5')), 'r') as hf:
			features = np.array(hf.get('inception_features'))
	with h5py.File( join( data_dir, (split + '_image_id_list.h5')), 'r') as hf:
		image_id_list = np.array(hf.get('image_id_list'))
	return features, image_id_list

if __name__ == '__main__':
	prepare_training_data()
	# prepare_testing_data()