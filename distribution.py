import os
from sys import exit
from shutil import copyfile, rmtree
from random import shuffle
from datetime import datetime
from time import sleep
try:
	from matplotlib import pyplot as plt
	from matplotlib.ticker import MaxNLocator
	from torch import __version__ as torchVersion, device as torchDevice, load as torchLoad, max as torchMax, nn, no_grad, optim, save as torchSave
	from torch.cuda import is_available
	from torch.optim.lr_scheduler import StepLR
	from torch.utils.data import SubsetRandomSampler, DataLoader
	from torchvision import models, transforms
	from torchvision.datasets import ImageFolder
	from torchvision.models.resnet import resnet18, resnet34, resnet50, resnet101, resnet152
	from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
	from seaborn import heatmap
	from tqdm import tqdm
except Exception as e:
	print("Failed importing related libraries. Details are as follows. \n{0}\n\nPlease press the enter key to exit. ".format(e))
	input()
	exit(-1)
try:
	os.chdir(os.path.abspath(os.path.dirname(__file__)))
except:
	pass
EXIT_SUCCESS = 0
EXIT_FAILURE = 1
EOF = (-1)
useNet = 18
maxEpoch = 2
batchSize = 64
initialLearningRate = 0.001
dataSetPath = "dataSet"
randomSplit = True
splitRate = 0.8
trainingSetPath = None
testingSetPath = None
isShuffle = True
modelFilePath = "modelcatdog.pth"
trainingLogFilePath = "training.log"
sampling = 100
testingLogFilePath = "testing.log"
performanceFigureFilePathFormat = "performanceFigure" + os.sep + "{0}.png"
dpi = 1200
pauseTime = 10 # to have a rest, e.g. 10s, 30s, 60s, 0 for no rests


# Class #
class Net(nn.Module):
	def __init__(self, model):
		super(Net, self).__init__()
		self.conv1 = nn.Conv2d(3, 64, kernel_size = 5, stride = 2, padding = 2, bias = False)
		self.resnet_layer = nn.Sequential(*list(model.children())[1:-1]) # Remove the last layer of the model
		if useNet in (18, 34):
			self.Linear_layer = nn.Linear(512, 8) # Add a full connection layer with modified parameters
		elif useNet in (50, 101, 152):
			self.Linear_layer = nn.Linear(2048, 32) # Add a full connection layer with modified parameters
	def forward(self, x):
		x = self.conv1(x)
		x = self.resnet_layer(x)
		x = x.view(x.size(0), -1)
		x = self.Linear_layer(x)
		return x

class ResNet:
	netSeries = (18, 34, 50, 101, 152)
	netNames = ("ResNet18", "ResNet34", "ResNet50", "ResNet101", "ResNet152")
	netDict = {18:resnet18, 34:resnet34, 50:resnet50, 101:resnet101, 152:resnet152}
	weightsDict = {18:models.ResNet18_Weights.DEFAULT, 34:models.ResNet34_Weights.DEFAULT, 50:models.ResNet50_Weights.DEFAULT, 101:models.ResNet101_Weights.DEFAULT, 152:models.ResNet152_Weights.DEFAULT}
	netEpochDict = {18:24, 34:14, 50:18, 101:8, 152:12} # Pre-set max training epochs
	folderInitialComments = (".", "#", "%", "//")
	def __init__(													\
		self:object, useNet:int|str, maxEpoch:int = None, batchSize:int = 16, initialLearningRate:float = 0.001, dataSetPath:str = "dataSet", 	\
		randomSplit:bool = True, splitRate:float = 0.8, trainingSetPath:str = "trainingSet", testingSetPath:str = "testingSet", isShuffle = True, 	\
		modelFilePath:str = "model.pth", trainingLogFilePath:str = "training.log", sampling:int = 100, testingLogFilePath:str = "testing.log", 	\
		performanceFigureFilePathFormat:str = "performanceFigure" + os.sep + "{0}.png", dpi:int = 1200, pauseTime:int = 10			\
	) -> object:
		# useNet #
		if isinstance(useNet, int) and useNet in ResNet.netSeries:
			self.useNet = useNet
		elif isinstance(useNet, str) and useNet in ResNet.netNames:
			self.useNet = ResNet.netSeries[ResNet.netNames.index(useNet)]
		else:
			print("Unknown ResNet is selected. It is defaulted to {0}. ".format(ResNet.netNames[0]))
			self.useNet = ResNet.netSeries[0]
		
		# name #
		self.name = "ResNet" + str(useNet)
		
		# net #
		if ResNet.compareVersion(torchVersion, "0.13") == -1: # torchVersion < "0.13"
			self.net = ResNet.netDict[useNet](pretrained = True)
		else:
			self.net = ResNet.netDict[useNet](weights = ResNet.weightsDict[useNet])
		
		# maxEpoch #
		if isinstance(maxEpoch, int) and maxEpoch > 0:
			self.maxEpoch = maxEpoch
		else:
			print("Unknown maximum epoch is specified. It is defaulted to {0}. ".format(ResNet.netEpochDict[self.useNet]))
			self.maxEpoch = ResNet.netEpochDict[self.useNet]
		
		# batchSize #
		if isinstance(batchSize, int) and batchSize > 0:
			self.batchSize = batchSize
		else:
			print("Unknown batch size is specified. It is defaulted to 16. ")
			self.batchSize = 16
		
		# initialLearningRate #
		if isinstance(initialLearningRate, float) and 0 < initialLearningRate < 1:
			self.initialLearningRate = initialLearningRate
		
		# randomSplit #
		if isinstance(randomSplit, bool):
			self.randomSplit = randomSplit
		else:
			print("Unknown shuffling option is specified. It is defaulted to be on. ")
			self.randomSplit = True
		
		# dataSetPath #
		if self.randomSplit:
			if isinstance(dataSetPath, str) and os.path.isdir(dataSetPath):
				self.dataSetPath = dataSetPath
			else:
				self.dataSetPath = None
				while not self.dataSetPath:
					print("The path to the overall dataset is not specified or does not exist. Please enter a new path: ")
					tmpPath = input("")
					if tmpPath and os.path.isdir(tmpPath):
						self.dataSetPath = tmpPath
					else:
						self.dataSetPath = "."
			if isinstance(splitRate, float) and 0 < splitRate < 1:
				self.splitRate = splitRate
			else:
				print("Unknown or illegal split rate is specified. It is defaulted to 0.8. ")
				self.splitRate = 0.8
		else:	
			if isinstance(trainingSetPath, str) and os.path.isdir(trainingSetPath):
				self.trainingSetPath = trainingSetPath
			else:
				self.trainingSetPath = None
				while not self.trainingSetPath:
					print("The path to the training set is not specified. Please enter a new path: ")
					tmpPath = input("")
					if tmpPath and os.path.isdir(tmpPath):
						self.trainingSetPath = tmpPath
					else:
						self.trainingSetPath = "."
			if isinstance(testingSetPath, str) and os.path.isdir(testingSetPath):
				self.testingSetPath = testingSetPath
			else:
				self.testingSetPath = None
				while not self.testingSetPath:
					print("The path to the testing set is not specified. Please enter a new path: ")
					tmpPath = input("")
					if tmpPath and os.path.isdir(tmpPath):
						self.testingSetPath = tmpPath
					else:
						self.testingSetPath = "."
		
		# isShuffle #
		if isinstance(isShuffle, bool):
			self.isShuffle = isShuffle
		else:
			print("Unknown shuffling option is specified. It is defaulted to be on. ")
			self.isShuffle = True
		
		# modelFilePath #
		self.modelFilePath = modelFilePath
		
		# trainingLogFilePath #
		self.trainingLogFilePath = trainingLogFilePath
		
		# sampling #
		if isinstance(sampling, int) and sampling > 0:
			self.sampling = sampling
		else:
			print("Unknown or illegal sampling is specified. It is defaulted to 100. ")
			self.sampling = 100

		# testingLogFilePath #
		self.testingLogFilePath = testingLogFilePath
		
		# performanceFigureFilePathFormat #
		self.performanceFigureFilePathFormat = performanceFigureFilePathFormat
		
		# dpi #
		if isinstance(dpi, int) and dpi > 300:
			self.dpi = dpi
		else:
			print("Unknown or illegal dpi is specified. It is defaulted to 1200. ")
			self.dpi = 1200
		
		# pauseTime #
		if (isinstance(pauseTime, int) or isinstance(pauseTime, float)) and pauseTime > 0:
			self.pauseTime = pauseTime
		else:
			print("Unknown or illegal pausing time is specified. It is defaulted to 0. ")
			self.pauseTime = 0
	
	# Load #
	def getClasses(self:object, targetSetPath:str, targetSetName:str) -> tuple:
		if os.path.isdir(targetSetPath):
			classes = []
			for d in os.listdir(targetSetPath):
				if os.path.isdir(os.path.join(targetSetPath, d)):
					flagSkip = False
					for folderInitialComment in ResNet.folderInitialComments:
						if d.startswith(folderInitialComment): # skipped commented folders
							flagSkip = True
							break
					if flagSkip:
						continue
					else:
						classes.append(d)
			classCount = len(classes)
			if classCount > 1:
				print("There are {0} {1} classes in total listed as follows. \n{2}".format(classCount, targetSetName, classes))
			else:
				print("The count of {0} classes should be no smaller than 2. ".format(targetSetName))
			return (classCount, classes)
		else:
			print("Read data failed. The following folder does not exist. \n{0}".format(targetSetPath))
			return (0, [])
	def load(self:object) -> None:
		if self.randomSplit:
			self.dataClassCount, self.dataClasses = self.getClasses(self.dataSetPath, "data")
		else:
			self.trainingClassCount, self.trainingClasses = self.getClasses(self.trainingSetPath, "training")
			self.testingClassCount, self.testingClasses = self.getClasses(self.testingSetPath, "testing")
		if useNet in ResNet.netSeries[:2]:
			transformPIL = transforms.RandomApply([transforms.RandomHorizontalFlip(p = 1), transforms.RandomVerticalFlip(p = 1), transforms.RandomRotation(45)], p = 0.5)
		else:
			transformPIL = transforms.RandomApply([transforms.RandomHorizontalFlip(p = 1), transforms.RandomVerticalFlip(p = 1), transforms.RandomRotation(45)], p = 0.4)
		transformTrain = transforms.Compose(
			[
				transforms.Resize((256, 256)),  # resize to 256x256
				transformPIL, 
				transforms.RandomCrop((225, 225)),  # randomly cut to 224x224
				transforms.RandomHorizontalFlip(),  # flip horizontally
				transforms.ToTensor(), 
				transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.2225)) # Normalization, the value is given by Imagenet
			]
		)
		transformTest = transforms.Compose(
			[
				transforms.Resize((224, 224)), 
				transforms.ToTensor(), 
				transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
			]
		)
		
		if self.randomSplit:
			dataset = ImageFolder(root = self.dataSetPath, transform = transformTrain)
			dataLoader = DataLoader(dataset, batch_size = batchSize, shuffle = self.isShuffle, num_workers = 4)
			indexList = list(range(len(dataset)))
			if self.isShuffle:
				shuffle(indexList)
			trainingIdx = indexList[:int(len(dataset) * self.splitRate)]
			testingIdx = indexList[int(len(dataset) * self.splitRate):]
			trainingSampler = SubsetRandomSampler(trainingIdx)
			testingSampler = SubsetRandomSampler(testingIdx)
			self.trainingLoader = DataLoader(dataset, batch_size = self.batchSize, sampler = trainingSampler, num_workers = 4)
			self.testingLoader = DataLoader(dataset, batch_size = self.batchSize, sampler = testingSampler, num_workers = 4)
			print("There are {0} data in total, {1} of which are for training and {2} for testing. ".format(len(dataset), len(trainingIdx), len(testingIdx)))
		else:
			trainingSet = ImageFolder(root = self.trainingSetPath, transform = transformTrain)
			testingSet = ImageFolder(root = self.testingSetPath, transform = transformTest)
			trainingIdx = list(range(len(trainingSet)))
			testingIdx = list(range(len(testingSet)))
			if self.isShuffle:
				shuffle(trainingIdx)
				shuffle(testingIdx)
			trainingSampler = SubsetRandomSampler(trainingIdx)
			testingSampler = SubsetRandomSampler(testingIdx)
			self.trainingLoader = DataLoader(trainingSet, batch_size = self.batchSize, sampler = trainingSampler, num_workers = 4)
			self.testingLoader = DataLoader(testingSet, batch_size = self.batchSize, sampler = testingSampler, num_workers = 4)
			print("There are {0} data in total, {1} of which are for training and {2} for testing. ".format(len(trainingSet) + len(testingSet), len(trainingSet), len(testingSet)))
	
	# Evaluate #
	def getDetailedAccuracy(self:object, output:object, label:object) -> float:
		total = output.shape[0]
		_, pred_label = output.max(1)
		num_correct = (pred_label == label).sum().item()
		return num_correct / total if total else float("nan")
	def getGeneralAccuracy(self:object, model:object, device:object) -> float:
		model.eval()
		total = 0
		correct = 0
		with no_grad():
			for batchIdx, (img, label) in enumerate(self.testingLoader):
				image = img.to(device)
				label = label.to(device)
				out = model(image)
				_, predicted = torchMax(out.data, 1)
				total += image.size(0)
				correct += predicted.data.eq(label.data).cpu().sum()
		return (1.0 * correct.numpy()) / total if total else float("nan")
	def log(self:object, content:str, outputFp:str, mode = "w", encoding:str = "utf-8") -> bool:
		if not content or not outputFp:
			return None
		elif ResNet.handleFolder(os.path.split(outputFp)[0]):
			try:
				with open(outputFp, mode = mode, encoding = encoding) as f:
					f.write(str(content))
				print("Log to \"{0}\" successfully. ".format(outputFp))
				return True
			except Exception as e:
				print("Log to \"{0}\" failed. Details are as follows. \n{1}".format(outputFp, e))
				return False
		else:
			print("Log to \"{0}\" failed since the parent folder is not created successfully. ".format(outputFp))
			return False
	def draw(self:object, x:list, y:list, color:str = None, marker:str = None, legend:list = None, title:str = None, xlabel:str = None, ylabel:str = None, isInteger:bool = True, savefigPath:str = None, dpi:int = 1200) -> bool:
		if color and marker:
			plt.plot(x, y, color = color, marker = marker)
		elif color:
			plt.plot(x, y, color = color)
		elif marker:
			plt.plot(x, y, marker = marker)
		else:
			plt.plot(x, y)
		plt.rcParams["figure.dpi"] = 300
		plt.rcParams["savefig.dpi"] = 300
		plt.rcParams["font.family"] = "Times New Roman"
		if legend:
			plt.legend(legend)
		if title:
			plt.title(title)
		plt.gca().xaxis.set_major_locator(MaxNLocator(integer = isInteger))
		if xlabel:
			plt.xlabel(xlabel)
		if ylabel:
			plt.ylabel(ylabel)
		plt.rcParams["figure.dpi"] = dpi
		plt.rcParams["savefig.dpi"] = dpi
		if savefigPath:
			if ResNet.handleFolder(os.path.split(savefigPath)[0]):
				try:
					plt.savefig(savefigPath)
					plt.close()
					print("Save the figure to \"{0}\" successfully. ".format(savefigPath))
					return True
				except Exception as e:
					print("Failed saving the figure to \"{0}\". Details are as follows. \n{1}".format(savefigPath, e))
					return False
			else:
				print("Failed saving the figure to \"{0}\" since the parent folder is not created successfully. ".format(savefigPath))
				plt.show()
				plt.close()
				return False
		else:
			plt.show()
			plt.close()
			return True
	
	# Train #
	def train(self:object) -> None:
		device = torchDevice("cuda" if is_available() else "cpu") # GPU performs better than CPU
		print("Getting ready to train. Device currently used is {0}. ".format(device))
		model = Net(self.net).to(device)
		#optimizer = optim.SGD(model.parameters(), lr = self.initialLearningRate, momentum = 0.8, weight_decay = 3e-3)
		optimizer = optim.Adam(model.parameters(), lr = self.initialLearningRate, betas = (0.9, 0.999), eps = 1e-8, weight_decay = 5e-4, amsgrad = True)
		scheduler = StepLR(optimizer, step_size = 4, gamma = 0.3)
		criterion = nn.CrossEntropyLoss()
		trainingLoaderCount = len(self.trainingLoader)
		trainingGeneralAccuracy, trainingDetailedAccuracy, trainingGeneralLoss, trainingDetailedLoss = [], [], [], []
		samplingIdx = 0
		print("\n\nStart to train the model. \n")
		startTime = datetime.now()
		try:
			for epoch in range(1, self.maxEpoch + 1):
				trainingDetailedAccuracy.append([])
				trainingDetailedLoss.append([])
				print("Training epoch: {0}".format(epoch))
				if ResNet.compareVersion(torchVersion, "1.1") == -1: # torchVersion < "1.1"
					scheduler.step()
				model.train()
				for batchIdx, (img, label) in enumerate(self.trainingLoader):
					image = img.to(device)
					label = label.to(device)
					optimizer.zero_grad()
					out = model(image)
					loss = criterion(out, label)
					loss.backward()
					optimizer.step()
					accuracy = self.getDetailedAccuracy(out, label)
					lossValue = float(loss.mean().item())
					trainingDetailedAccuracy[-1].append(accuracy)
					trainingDetailedLoss[-1].append(lossValue)
					print("Epoch: {0} [{1}|{2}]  Accuracy: {3}  Loss: {4}".format(epoch, batchIdx + 1, trainingLoaderCount, accuracy, lossValue))
					samplingIdx = (samplingIdx + 1) % self.sampling
					if not samplingIdx:
						self.log("trainingGeneralAccuracy = {0}\n\ntrainingDetailedAccuracy = {1}\n\ntrainingGeneralLoss = {2}\n\ntrainingDetailedLoss = {3}".format(		\
							trainingGeneralAccuracy, trainingDetailedAccuracy, trainingGeneralLoss, trainingDetailedLoss), self.trainingLogFilePath			\
						)
				scheduler.step()
				generalAccuracy = self.getGeneralAccuracy(model, device)
				trainingGeneralAccuracy.append(generalAccuracy)
				generalLoss = sum(trainingDetailedLoss[-1]) / len(trainingDetailedLoss[-1])
				trainingGeneralLoss.append(generalLoss)
				self.log("trainingGeneralAccuracy = {0}\n\ntrainingDetailedAccuracy = {1}\n\ntrainingGeneralLoss = {2}\n\ntrainingDetailedLoss = {3}".format(		\
					trainingGeneralAccuracy, trainingDetailedAccuracy, trainingGeneralLoss, trainingDetailedLoss), self.trainingLogFilePath			\
				)
				print("Epoch: {0}  Accuracy: {1}  Loss: {2}\n".format(epoch, generalAccuracy, generalLoss))
				if epoch != self.maxEpoch:
					sleep(self.pauseTime)
		except KeyboardInterrupt:
			print("The training is interrupted by users. ")
		except Exception as e:
			print("Failed training the model. Details are as follows. \n{0}".format(e))
		endTime = datetime.now()
		if ResNet.handleFolder(os.path.split(self.modelFilePath)[0]):
			try:
				torchSave(model, self.modelFilePath) # save model
				print("The model is successfully save to \"{0}\". ".format(self.modelFilePath))
			except Exception as e:
				print("Failed saving the model to \"{0}\". Details are as follows. \n{1}".format(self.modelFilePath, e))
		else:
			print("Failed saving the model to \"{0}\" since the parent folder is not created successfully. ".format(self.modelFilePath))
			return False
		if trainingGeneralAccuracy:
			if 1 in trainingGeneralAccuracy:
				print("Maybe this model has over-fitted to your training set. ")
			print("trainingGeneralAccuracy = {0}".format(trainingGeneralAccuracy))
			maxTrainingGeneralAccuracy = max(trainingGeneralAccuracy)
			maxTrainingGeneralAccuracyEpoch = trainingGeneralAccuracy.index(maxTrainingGeneralAccuracy)
			print(																\
				"Max accuracy approached at epoch {0} is {1}%. \nThe overall time consumption is {2:.6f}s. \nThe time cost to this accuracy is {3:.6f}s".format(		\
					maxTrainingGeneralAccuracyEpoch, maxTrainingGeneralAccuracy, (endTime - startTime).total_seconds() - pauseTime * (self.maxEpoch - 1), 	\
					((endTime - startTime).total_seconds() - pauseTime * (self.maxEpoch - 1)) * maxTrainingGeneralAccuracyEpoch / self.maxEpoch		\
				)															\
			)
			self.draw(																\
				[i for i in range(1, len(trainingGeneralAccuracy) + 1)], trainingGeneralAccuracy, color = "orange", marker = "x", legend = ["Accuracy"], title = None, 		\
				xlabel = "Epoch", ylabel = "Accuracy", savefigPath = self.performanceFigureFilePathFormat.format("trainingGeneralAccuracy"), dpi = self.dpi		\
			)
		else:
			print("No training general accuracy values are collected. ")
		if trainingDetailedAccuracy and trainingDetailedAccuracy[0]:
			flatDetailedAccuracy = []
			for accuracyList in trainingDetailedAccuracy:
				flatDetailedAccuracy += accuracyList
			self.draw(																\
				[i for i in range(1, len(flatDetailedAccuracy) + 1)], flatDetailedAccuracy, color = "orange", marker = None, legend = ["Accuracy"], title = None, 		\
				xlabel = "Sample", ylabel = "Accuracy", savefigPath = self.performanceFigureFilePathFormat.format("trainingDetailedAccuracy"), dpi = self.dpi		\
			)
		else:
			print("No training detailed accuracy values are collected. ")
		if trainingGeneralLoss:
			self.draw(															\
				[i for i in range(1, len(trainingGeneralLoss) + 1)], trainingGeneralLoss, color = "orange", marker = "x", legend = ["Loss"], title = None, 		\
				xlabel = "Epoch", ylabel = "Loss", savefigPath = self.performanceFigureFilePathFormat.format("trainingGeneralLoss"), dpi = self.dpi		\
			)
		else:
			print("No training general loss values are collected. ")
		if trainingDetailedLoss and trainingDetailedLoss[0]:
			flatDetailedLoss = []
			for lossList in trainingDetailedLoss:
				flatDetailedLoss += lossList
			self.draw(															\
				[i for i in range(1, len(flatDetailedLoss) + 1)], flatDetailedLoss, color = "orange", marker = None, legend = ["Loss"], title = None, 			\
				xlabel = "Sample", ylabel = "Loss", savefigPath = self.performanceFigureFilePathFormat.format("trainingDetailedLoss"), dpi = self.dpi		\
			)
		else:
			print("No training detailed loss values are collected. ")
		print("\nThe training is finished. \n\n")
	
	# Test #
	def test(self:object) -> None:
		if not os.path.isfile(self.modelFilePath):
			print("The model file does not exist. The testing is failed due to no models loaded. ")
			return
		print("Start to test the model. ")
		model = torchLoad(self.modelFilePath) if is_available() else torchLoad(self.modelFilePath, map_location = "cpu")
		model.cpu()
		classes = self.dataClasses if self.randomSplit else self.trainingClasses
		classCount = len(classes)
		testingReal = []
		testingPredicted = []
		with no_grad():
			for data in tqdm(self.testingLoader, ncols = 100):
				images, labels = data
				outputs = model(images)
				_, predicted = torchMax(outputs, 1)
				testingReal += [labels[i].item() for i in range(len(labels))]
				testingPredicted += [predicted[i].item() for i in range(len(predicted))]
		try:
			testingConfusionMatrix = confusion_matrix(testingReal, testingPredicted)
			testingAccuracyScore = accuracy_score(testingReal, testingPredicted)
			testingF1Score = f1_score(testingReal, testingPredicted, average = "micro")
			testingPrecisionScore = precision_score(testingReal, testingPredicted, average = "micro")
			testingRecallScore = recall_score(testingReal, testingPredicted, average = "micro")
			self.log(																		\
				"Testing confusion matrix: \n{0}\n\nTesting accuracy score: {1}\nTesting f1 score: {2}\nTesting precision score: {3}\nTesting recall score: {4}\n\nLabels: \n{5}".format(		\
					testingConfusionMatrix, testingAccuracyScore, testingF1Score, testingPrecisionScore, testingRecallScore, 							\
					"\n".join(["{0}\t{1}".format(i, label) for i, label in enumerate(classes)])										\
				), 																	\
				testingLogFilePath																\
			)
			print(																\
				"Testing confusion matrix: \n{0}\n\nTesting accuracy score: {1}\nTesting f1 score: {2}\nTesting precision score: {3}\nTesting recall score: {4}".format(	\
					testingConfusionMatrix, testingAccuracyScore, testingF1Score, testingPrecisionScore, testingRecallScore, 					\
				)															\
			)
			plt.rcParams["figure.dpi"] = 300
			plt.rcParams["savefig.dpi"] = 300
			heatmap(testingConfusionMatrix, annot = True, fmt = "d", cmap = "BuPu")
			plt.xlabel("Predicted")
			plt.ylabel("Real")
			plt.rcParams["figure.dpi"] = self.dpi
			plt.rcParams["savefig.dpi"] = self.dpi
			savePath = self.performanceFigureFilePathFormat.format("testingConfusionMatrix")
			if ResNet.handleFolder(os.path.split(savePath)[0]):
				plt.savefig(savePath)
				print("Save the testing heatmap to \"{0}\" successfully. ".format(savePath))
			else:
				plt.show()
		except Exception as e:
			print("Failed generating the confusion matrix. Details are as follows. \n{0}".format(e))
		finally:
			plt.close()
		print("The testing is finished. ")
	
	# Static #
	@staticmethod
	def compareVersion(version1:str, version2:str) -> int:
		if version1 == version2:
			return 0
		v1 = version1.split(".")
		v2 = version2.split(".")
		while v1 and v2:
			x = v1.pop(0)
			y = v2.pop(0)
			try:
				x = int(x)
				y = int(y)
			except:
				pass
			if x > y:
				return 1
			else:
				return -1
		if v1:
			return 1
		elif v2:
			return -1
		else:
			return 0
	@staticmethod
	def handleFolder(folder:str) -> bool:
		if folder in ("", ".", "./"):
			return True
		elif os.path.exists(folder):
			return os.path.isdir(folder)
		else:
			try:
				os.makedirs(folder)
				return True
			except:
				return False


# Function #
def main() -> int:
	resNet = ResNet(
		useNet = useNet, maxEpoch = maxEpoch, batchSize = batchSize, initialLearningRate = initialLearningRate, 	\
		dataSetPath = dataSetPath, randomSplit = randomSplit, splitRate = splitRate, trainingSetPath = trainingSetPath, 	\
		testingSetPath = testingSetPath, isShuffle = isShuffle, 	modelFilePath = modelFilePath, 			\
		trainingLogFilePath = trainingLogFilePath, sampling = sampling, testingLogFilePath = testingLogFilePath, 		\
		performanceFigureFilePathFormat = performanceFigureFilePathFormat, dpi = dpi, pauseTime = pauseTime		\
	)
	resNet.load()
	resNet.train()
	resNet.test()
	print("\nAll the procedures are finished. Please press the enter key to exit. \n")
	input()
	return EXIT_SUCCESS



if __name__ == "__main__":
	exit(main())