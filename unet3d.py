from tensorflow.keras.layers import ( Conv3D, BatchNormalization, 
                                    Activation, MaxPool3D, Conv3DTranspose,
                                    Concatenate, Input)

from tensorflow.keras.models import Model

def convSet(inputs, numFilters):
    x = Conv3D(numFilters, 3, padding="same")(inputs)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    
    x = Conv3D(numFilters, 3, padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    return x

def encoder(inputs, numFilters):
    x = convSet(inputs, numFilters)
    p = MaxPool3D((2,2,2))(x)
    return x, p

def decoder(input, features, numFilters):
    x = Conv3DTranspose(numFilters,(2,2,2), strides =2, padding="same")(input)
    x = Concatenate()([x, features])
    x = convSet(x, numFilters)
    return x

def unet3d(inputShape:tuple):
    inputs = Input(shape = (inputShape))

    s1, p1 = encoder(inputs, 16)
    s2, p2 = encoder(p1, 32)
    s3, p3 = encoder(p2, 64)
    s4, p4 = encoder(p3, 128)
    
    b1 = convSet(p4, 256)

    d1 = decoder(b1, s4, 128)
    d2 = decoder(d1, s3, 64)
    d3 = decoder(d2, s2, 32)
    d4 = decoder(d3, s1, 16)

    outputs = Conv3D(4, 1, padding="same", activation="softmax")(d4)
    model = Model(inputs, outputs, name="UNet3d")
    return model

if __name__ == "__main__":
    unet3d()
