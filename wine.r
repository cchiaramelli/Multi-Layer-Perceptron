source('mlp.r')

preprocess <- function(){

	data = as.matrix(read.table("wine.data", sep=","))

	#Normalize
	for (i in 2:ncol(data)){
		data[,i] = (data[,i]-min(data[,i])) / (max(data[,i])-min(data[,i]))
	}

	#Put the class columns in the end
	data = cbind(data[,2:ncol(data)], class.ind(data[,1]))

	#randomize the dataset
	data = data[sample(nrow(data)),]

	return (data)
}

wine.run <- function(n.hyperplanes=7, traindataset.proportion=0.75, eta=0.1, threshold=1e-3, max.iterations=10000){

	data = preprocess()

	traindataset.proportion = min(1, traindataset.proportion)
	traindataset.size = nrow(data)*traindataset.proportion

	#Separate between train and test datasets
	traindataset = data[1:traindataset.size,]
	testdataset = data[-(1:traindataset.size),]

	#Initialize and train the model
	model = generate.model(13, n.hyperplanes, 3)
	model = train(model=model, trainset = traindataset, threshold=threshold, n.iterations=max.iterations, eta=eta)

	#Test the trained model
	test(model, testdataset)
}