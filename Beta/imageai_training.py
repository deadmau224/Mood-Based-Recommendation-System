from imageai.Prediction.Custom import ModelTraining
import os

os.environ['KMP_DUPLICATE_LIB_OK']='True' #to fix the overflow issue

model_trainer = ModelTraining()
model_trainer.setModelTypeAsResNet()
model_trainer.setDataDirectory("/Users/digvijayghotane/Desktop/Projects/mood_based_recommendation_system/Beta/images")
model_trainer.trainModel(num_objects=5, num_experiments=10, enhance_data=True, batch_size=32, show_network_summary=True)