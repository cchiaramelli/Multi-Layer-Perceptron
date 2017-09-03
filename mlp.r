

generate.model <- function (input.size, hidden.size, output.size){
	
	ret = list()
	ret$input.size = input.size
	ret$hidden.size = hidden.size
	ret$output.size = output.size

	# Weigths:
	# neuron1 [w1, w2, w3,...,theta]
	# neuron2 [w1, w2, w3,...,theta]
	# ...

	ret$w_h = matrix(runif((input.size+1)*hidden.size, -1, 1),hidden.size)
	ret$w_o = matrix(runif((hidden.size+1)*output.size, -1, 1),output.size)

	return (ret)
}

#Applies the sigmoid function
sigmoid.function <- function(x){
	return (1/(1+exp(1)^(-x)))
}

#Applies the model, given the input
forward <- function (model, input){
	
	ret = list()

	ret$net_h = as.matrix(model$w_h) %*% as.numeric(append(input,1))
	ret$f_h = sigmoid.function( ret$net_h )

	ret$net_o = as.matrix(model$w_o) %*% as.numeric(append(ret$f_h,1))
	ret$f_o = sigmoid.function( ret$net_o )

	return (ret)
}


train <- function (model, trainset, threshold=1e-3, eta=0.1, n.iterations=10000){

	#desiredSet is the desired values (Answers)
	desiredSet = as.matrix(trainset[,-(1:model$input.size)])

	#forwardSet is the rest of the data
	forwardSet = as.matrix(trainset[,1:model$input.size])

	avg_sqr_error=threshold
	while(avg_sqr_error >= threshold && n.iterations > 0){
		avg_sqr_error=0
		n.iterations= n.iterations-1

		for (input in 1:nrow(trainset)){

			#Calculate the output with the current weigths
			output = forward(model=model, input=forwardSet[input,])

			#Calculate the error for each output neuron
			error = as.numeric(desiredSet[input,]) - as.numeric(output$f_o)
			#cat ("e: ", error, "\n")

			#Calculate the delta_o
			delta_o = error * (output$f_o)*(1-output$f_o)

			#Calculate the delta_h
			delta_h = colSums(matrix(as.numeric(delta_o)*model$w_o[,1:model$hidden.size], nrow=model$output.size)) * (output$f_h)*(1-output$f_h)
			
			#Update the weigths
			model$w_o = model$w_o + eta * delta_o %*% t(append(output$f_h,1))
			model$w_h = model$w_h + eta * delta_h %*% t(append(as.numeric(forwardSet[input,]),1))

			#Calculate the average squared error of each test
			avg_sqr_error=avg_sqr_error + sum(error * error)
		}
		avg_sqr_error=avg_sqr_error/nrow(trainset)

		if (n.iterations%%100==0){
			cat("Squared error: ", avg_sqr_error, "\n")
		}
	}

	return (model)
}

#Run a test dataset
test <- function (model, testdataset){

	right = 0
	for (i in 1:nrow(testdataset)){

		output = forward(model, testdataset[i, 1:model$input.size])

		#Rounds the binary result of each neuron
		rounded = as.numeric(round(output$f_o))

		expected = testdataset[i,(model$input+1):ncol(testdataset)]

		#Count the right guesses
		if (all.equal(rounded, as.numeric(expected))==TRUE){
			right = right+1
		}
	}

	cat ("Accuracy: ", 100*right/nrow(testdataset), "% of ", nrow(testdataset), " tests\n")

}