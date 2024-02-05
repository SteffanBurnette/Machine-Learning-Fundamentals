def pred_and_plot(model, filename, class_names=class_names):
  """ Imports a image located at filename, makes a prediction with model and
  plots the image with the predicted class as the title
  """
  #Import the target image and preprocess it
  img = load_and_prep_img(filename)

  #Make a prediction
  pred = model.predict(tf.expand_dims(img, axis=0))

  #Add in logic for predictions since a softmax function returns a nested list of multiple predictions
  #Unlike a sigmoid function which only returns the predicted value
  if len(pred)> 1: #Handles multiclass images logic
    pred_class = class_names[tf.argmax(pred[0])] #Gets the class name of the highest prediction value
  else:  #Handles binary class images logic
    #Get the predicted class
    pred_class = class_names[int(tf.round(pred[0]))]

  #Plot the image and predicted class
  plt.imshow(img)
  plt.title(f"Prediction: {pred_class}")
  plt.axis(False);