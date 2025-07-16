def predict(new_input):
    normalized_input = normalise_input([new_input])
    NN(normalized_input[0])
    prediction = NN.get_output_value()
    return denormalise_output(prediction) 
