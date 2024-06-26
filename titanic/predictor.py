import os

import tensorflow_decision_forests as tfdf
import numpy as np
import pandas as pd
import tensorflow as tf
import tf_keras
import math

def main():
	pd_train = pd.read_csv('data/train.csv');
	pd_test = pd.read_csv('data/test.csv');

	train = tfdf.keras.pd_dataframe_to_tf_dataset(pd_train,label='Survived');
	test = tfdf.keras.pd_dataframe_to_tf_dataset(pd_test);

	model = tfdf.keras.RandomForestModel();

	model.fit(train);

	with open("plotRandom.html", "w") as f: 
		f.write(tfdf.model_plotter.plot_model(model))

	model.summary();
	model.make_inspector().variable_importances()

	result = model.predict(test);
	testSurvived = [1 if p>=0.5 else 0 for p in result];

	output = pd.DataFrame({'PassengerId': pd_test.PassengerId, 'Survived': testSurvived});
	output.to_csv('data/randomSubmission.csv',index=False);
	print("submission saved!");

	model2 = tfdf.keras.GradientBoostedTreesModel();
	model2.fit(train);
	with open("plotGrad.html", "w") as f: 
		f.write(tfdf.model_plotter.plot_model(model2))

	model2.summary();
	model2.make_inspector().variable_importances()

	result2 = model2.predict(test);
	testSurvived2 = [1 if p>=0.5 else 0 for p in result2];

	print("Models differed on the predictions of: ",np.sum(np.abs(np.array(testSurvived) - np.array(testSurvived2))))

	output2 = pd.DataFrame({'PassengerId': pd_test.PassengerId, 'Survived': testSurvived2});
	output2.to_csv('data/gradSubmission.csv',index=False);
	print("submission saved!");

	model3 = tfdf.keras.GradientBoostedTreesModel(hyperparameter_template="benchmark_rank1");
	model3.fit(train);
	with open("plotGrad2.html", "w") as f: 
		f.write(tfdf.model_plotter.plot_model(model3))

	model3.summary();
	model3.make_inspector().variable_importances()

	result3 = model3.predict(test);
	testSurvived3 = [1 if p>=0.5 else 0 for p in result3];

	print("Boosted Models differed on the predictions of: ",np.sum(np.abs(np.array(testSurvived2) - np.array(testSurvived3))))

	output3 = pd.DataFrame({'PassengerId': pd_test.PassengerId, 'Survived': testSurvived3});
	output3.to_csv('data/gradSubmissionHyper.csv',index=False);
	print("submission saved!");

	




if __name__ == "__main__":
	main();