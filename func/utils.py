def print_model_summary(model, input_shape):
    model.build(input_shape=input_shape)
    model.summary()
    for layer in model.layers:
        print(f"Layer: {layer.name}, Trainable: {layer.trainable}")
