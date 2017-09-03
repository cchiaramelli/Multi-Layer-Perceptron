source('mlp.r')

xor <- function(model=NA, n.hyperplanes=2, eta=0.1, threshold=1e-3, max.iterations=10000){

	#The XOR table
	testdataset = matrix(c(0,0,1,1,
							0,1,0,1,
							0,1,1,0),ncol=3)

	#Trains if needed
	if (is.na(model)){
		#XOR is so simple, that the traindataset=testdataset
		traindataset = testdataset
		model = generate.model(2,n.hyperplanes,1)
		model = train(model=model, trainset=traindataset, eta=eta, threshold=threshold, n.iterations=max.iterations)
	}

	#Run the testdataset
	test(model, testdataset)
}