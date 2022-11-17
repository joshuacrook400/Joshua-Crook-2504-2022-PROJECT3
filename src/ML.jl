using Flux, Statistics, Random, StatsBase, Plots, LinearAlgebra
using Flux: params, onehotbatch, crossentropy, update!, binarycrossentropy



Random.seed!(0)


"""
Returns the accuracy and confusionMatrix of a linear training model
"""
function linear_one_vs_all(train_data, test_data, class_num)

    train_imgs = train_data[1];
    train_labels = train_data[2];

    test_imgs = test_data[1];
    test_labels = test_data[2];

    n_train, n_test = length(train_labels), length(test_labels)
    

    X = vcat([vec(train_imgs[:,:,k])' for k in 1:n_train]...);

    A = [ones(n_train) X];
    Adag = pinv(A);
   
    tfPM(x) = x ? +1 : -1
    yDat(k) = tfPM.(onehotbatch(train_labels, 0:class_num)'[:,k+1])
    betas = [Adag*yDat(k) for k in 0:class_num]; #this is the trained model (a list of 10 beta coeff vectors)

    linear_classify(square_image) = argmax([([1 ; vec(square_image)])'*betas[k] for k in 1:10]) - 1


    predictions = [linear_classify(test_imgs[:,:,k]) for k in 1:n_test]
    confusionMatrix = [sum((predictions .== i) .& (test_labels .== j)) for i in 0:class_num, j in 0:class_num]
    acc = sum(diag(confusionMatrix))/n_test

    return confusionMatrix, acc
end

"""
Returns the accuracy and confusionMatrix of a linear training model comparing data 1 against 2
@param X_1 Matrix of all training and test images for X_1
@param X_2 Matrix of all training and test images for X_2
@param X_1_size tuple conating number training values and test values for X_1
@param X_2_size tuple conating number training values and test values for X_2
"""
function linear_one_vs_one(X_1, X_2, X_1_size, X_2_size)

    X = vcat(X_1[1], X_2[1])
    Xt = vcat(X_1[2], X_2[2])

    X_labels = vcat(ones(X_1_size[1]), zeros(X_2_size[1]))

    n_train, n_test = size(X)[1], size(Xt)[1]
    
    
    A = [ones(n_train) X];
    Adag = pinv(A);
   
    tfPM(x) = x ? +1 : -1
    yDat(k) = tfPM.(onehotbatch(X_labels, 0:1)'[:,k+1])
    betas = [Adag*yDat(k) for k in 0:1];

    #if a value is less than 0 then it is classified as the first test
    linear_classify(square_image) = ((square_image' * betas[1][2:length(betas[1])]) + betas[1][1] .< 0) ? 1 : 0
   
    #testing values that should be 1
    sum = 0
    for i in 1:X_1_size[2]
        if linear_classify(Xt[i,:]) == 1
             sum += 1
        end
    end

    #testing values that should be 0
    for i in X_1_size[2] + 1:X_1_size[2] + X_2_size[2]
        if linear_classify(Xt[i,:]) == 0
            sum += 1
       end
    end
    
    #returns the accuracy as the value sum of all corretly identified value divided by total values
    return sum / n_test
end

sig(x) = 1/(1+float(MathConstants.e)^-x)
logistic_predict(img_vec, w, b) = sig.(w'*img_vec .+ b);
logistic_classifier(img_vec, w, b) = logistic_predict(img_vec, w, b) .> 0.5;


"""
Logs the accuracy of a logistic training model
"""
function logistic_one_vs_all(train_data, test_data)
    
    train_imgs = train_data[1];
    train_labels = train_data[2];

    test_imgs = test_data[1];
    test_labels = test_data[2];

    n_train, n_test = length(train_labels), length(test_labels)

    mini_batch_size = 1000


    #Initilize parameters
    w = randn(28*28)
    b = randn(1)

    
    loss(x, y) = binarycrossentropy(logistic_predict(x, w, b), y);

    loss_value = 0.0

    #training loop
    for epoch_num in 1:1

        
        counter = 1
        #Loop over mini-batches in epoch
        start_time = time_ns()
        for batch in Iterators.partition(1:n_train, mini_batch_size)
            batch_time = time_ns()
            

            C_b = 0
            C_w = 0
            
            
            for i in batch
                
                C_b = C_b .+ logistic_predict(train_imgs[i], w, b) .- train_labels[i]
                C_w = C_w .+ C_b * train_imgs[i]
                  
                
            end 

            
            b = b - [0.01 * (C_b ./ mini_batch_size)[1]] 
            w = w .- 0.01 * (C_w ./ mini_batch_size)
            
            
            

            batch_time_end = time_ns()
            println("Batch Number = $counter ($(round((batch_time_end-batch_time)/1e9,digits=2)) sec)")
           

            #modified code to prevent infinite recursion 
            if (counter == 5) 
                end_time = time_ns()
                println("Total elpased time after 5 batches $(round((end_time-start_time)/1e9,digits=2))")
                return nothing
            end

            counter += 1

        end
        
        end_time = time_ns()

        #record/display progress
        loss_value = loss(train_imgs', train_labels')
        println("Epoch = $epoch_num ($(round((end_time-start_time)/1e9,digits=2)) sec) Loss = $loss_value")
        
    end

    
    return w, b

end



function logistic_one_vs_all_auto_diff(train_data, test_data)
    
    train_imgs = train_data[1];
    train_labels = train_data[2];

    test_imgs = test_data[1];
    test_labels = test_data[2];

    n_train, n_test = length(train_labels), length(test_labels)

    mini_batch_size = 1000

    #Initilize parameters
    w = randn(28*28)
    b = randn(1)
    
    loss(x, y) = binarycrossentropy(logistic_predict(x, w, b), y);

    loss_value = 0.0

    #training loop
    for epoch_num in 1:20

        start_time = time_ns()

        for batch in Iterators.partition(1:n_train, mini_batch_size)

            gs = gradient(()->loss(train_imgs'[:,batch], train_labels[batch]'), params(w, b))
            b = b - η*gs[b]
            w = w - η*gs[w]
   
        end
        
        end_time = time_ns()

        #record/display progress
        loss_value = loss(train_imgs', train_labels')
        println("Epoch = $epoch_num ($(round((end_time-start_time)/1e9,digits=2)) sec) Loss = $loss_value")
        
    end

    
    return w, b

end
