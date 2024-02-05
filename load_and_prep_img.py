
#Create a function to import an image and resize it to be able to be used with our model
def load_and_prep_img(filename, img_shape=224):
  """
  Reads an image from filename, turns it into a tensor and reshapes it to (img_shape, img_shape, color_channels)
  """
  #Read in the image
  img = tf.io.read_file(filename)

  #Decode the read file into a tensor
  img = tf.image.decode_image(img)

  rgb_image = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
  img = rgb_image
  #Resize the image
  img = tf.image.resize(img, size = [img_shape, img_shape])
  #Rescale the image (get all values between 0 and 1)
  img=img/255.
  return img