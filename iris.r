require('nnet')
source('mlp.r')

preprocess <- function(){

	#Get the iris dataset
	data = iris

	#This separate the classes between binary identifiers, e.g.:
	# 0 -> 0 0 1
	# 1 -> 0 1 0
	# 2 -> 1 0 1

	data[,5:7] = class.ind(data[,5])

	for (i in 1:4){
		data[,i] = (data[,i] - min(data[,i]))/(max(data[,i])-min(data[,i]))
	}

	#randomize the dataset
	indexes = sample(nrow(data))

	return (data[indexes,])
}

iris.run <- function (n.hyperplanes=7, traindataset.proportion=0.75, eta=0.1, threshold=1e-3, max.iterations=10000){

	#Get the processed dataset
	data = preprocess()

	traindataset.proportion = min(1, traindataset.proportion)
	traindataset.size = nrow(data)*traindataset.proportion

	#Separate between train and test datasets	
	traindataset = data[1:traindataset.size,]
	testdataset = data[-(1:traindataset.size),]

	#Initialize and train the model
	model = generate.model(4, n.hyperplanes, 3)
	model = train(model=model, trainset = traindataset, threshold=threshold, n.iterations=max.iterations, eta=eta)

	#Test the trained model
	test(model, testdataset)

	return (model)
}