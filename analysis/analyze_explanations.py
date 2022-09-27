import os


model_info_list = [
	['who', 'regr', 'linear'], #regression
	['who', 'regr', 'ffnnA'],
	['who', 'regr', 'ffnnB'],
	['who', 'regr', 'ffnnC'],

	['heloc', 'classif', 'logistic'], #classification
	['heloc', 'classif', 'ffnnA'],
	['heloc', 'classif', 'ffnnB'],
	['heloc', 'classif', 'ffnnC']
	]

exp_folder = './experiments/results'

#####exp2: model recovery -- guiding principle
print('Performing Exp2, regression...')
os.system('python analysis/exp2_model_recov_regr.py')
print('Performing Exp2, classification...')
os.system('python analysis/exp2_model_recov_classif.py')



#####exp1: existing methods are instances of LFA framework
#####exp3: perturbation tests --- no free lunch theorem

#for each model, run exp1 and exp 3
for model_info in model_info_list:

		data_name = model_info[0]
		task = model_info[1]
		model_name = model_info[2]

		#exp1
		print('Performing Exp1...')
		os.system(f'python analysis/exp1_framework.py --exp_folder {exp_folder} --data_name {data_name} --model_name {model_name} --task {task}')

		#exp3
		#perturb bottom k features
		print('Performing Exp3, bottom-k features...')
		os.system(f'python analysis/exp3_perturb_tests.py --exp_folder {exp_folder} --data_name {data_name} --model_name {model_name} --task {task} --top_or_bottom_k bottomk')

		#perturb top k features
		print('Performing Exp3, top-k features...')
		os.system(f'python analysis/exp3_perturb_tests.py --exp_folder {exp_folder} --data_name {data_name} --model_name {model_name} --task {task} --top_or_bottom_k topk')

print('Analysis complete!')





