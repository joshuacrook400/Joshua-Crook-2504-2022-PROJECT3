using Flux
using Flux: binarycrossentropy #Loss function
using Flux: params #Used for automatic differenation (but in Project replace auto-diff with explicit gradient)
using FLUX: onehotbatch
using Random
using LinearAlgebra



Random.seed!(0)


"""
Returns the accuracy of a linear training model
"""
function train_linear_one_vs_all(train_imgs, train_labels)
    n_train = length(train_imgs)

    X = vcat([vec(train_imgs[:,:,k])' for k in 1:n_train]...);

    A = [ones(n_train) X];
    Adag = pinv(A);
   
    tfPM(x) = x ? +1 : -1
    yDat(k) = tfPM.(onehotbatch(train_labels_MINST,0:9)'[:,k+1])
    betas = [Adag*yDat(k) for k in 0:9]; #this is the trained model (a list of 10 beta coeff vectors)

    linear_classify(square_image) = argmax([([1 ; vec(square_image)])'*bets[k] for k in 1:10])-1
    predictions = [linear_classify(test_imgs[:,:,k]) for k in 1:n_test]
    confusionMatrix = [sum((predictions .== i) .& (test_labels .== j)) for i in 0:9, j in 0:9]


    return confusionMatrix
end






#functions
sig(x) = 1/(1+float(MathConstants.e)^-x)
logistic_predict(img_vec, w, b) = sig.(w'*img_vec .+ b);
logistic_classifier(img_vec, w, b) = logistic_predict(img_vec, w, b) .> 0.5; #Threhsold for predicting a positive sample


#η is the learning rate.  \eta + [TAB]
function train_logistic(train_data, train_labels, data_size ;num_epochs = 20, mini_batch_size = 100, η = 0.01)
    
    #Initilize parameters
    w = randn(28*28)
    b = randn(1)

    #As a loss function for training, We'll use the binary cross entropy 
    #which takes in a probability in [0,1]
    #and an actual label in {0,1}. The probability (of Ankle boot) 
    #in [0,1] is determined by the logistic model.
    loss(x, y) = binarycrossentropy(logistic_predict(x, w, b), y);
    
    loss_value = 0.0

    #Training loop
    for epoch_num in 1:num_epochs
        
        #Loop over mini-batches in epoch
        start_time = time_ns()
        for batch in Iterators.partition(1:data_size, mini_batch_size)
            
            gs = gradient(()->loss(train_data'[:,batch], train_labels[batch]'), params(w, b))
        
            b = b - η*gs[b]
            w = w - η*gs[w]
            
        end
        end_time = time_ns()
        

        #record/display progress
        loss_value = loss(train_data', train_labels')
        println("Epoch = $epoch_num ($(round((end_time-start_time)/1e9,digits=2)) sec) Loss = $loss_value")
        
    end
    return w, b
end

# Train model parameters
#w, b = train_logistic();