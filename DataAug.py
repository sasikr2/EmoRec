#assuming X_train X_test X_val are images already converted into numpy array in earlier processes like normalization
#from keras.preprocessing.image import img_to_array should be included


datagen = ImageDataGenerator( 
    rescale=1./255,
    rotation_range = 40,
    horizontal_flip = True,
    fill_mode = 'nearest')

testgen = ImageDataGenerator( 
    rescale=1./255
    )
datagen.fit(X_train)
batch_size = 64

#for printing 9 random data
for X_batch, y_batch in datagen.flow(X_train, y_train, batch_size=9,color_mode = 'grayscale', class_mode = 'categorical',target_size = (150,150)):
    for i in range(0, 9): 
        pyplot.axis('off') 
        pyplot.subplot(330 + 1 + i) 
        pyplot.imshow(X_batch[i].reshape(48, 48), cmap=pyplot.get_cmap('gray'))
    pyplot.axis('off') 
    pyplot.show() 
    break

 
train_flow = datagen.flow(X_train, y_train, batch_size=batch_size,color_mode = 'grayscale', class_mode = 'categorical',target_size = (150,150)) 
val_flow = testgen.flow(X_val, y_val, batch_size=batch_size,color_mode = 'grayscale', class_mode = 'categorical',target_size = (150,150)) 
test_flow = testgen.flow(X_test, y_test, batch_size=batch_size,color_mode = 'grayscale', class_mode = 'categorical',target_size = (150,150))