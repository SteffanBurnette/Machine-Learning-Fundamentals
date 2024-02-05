#Experimental
def load_and_prep_img(filename, img_shape= 224):
    # Load in the image
    img = tf.io.read_file(filename)
    img = tf.image.decode_image(img)

    # Check if the image is grayscale (if so, convert to RGB)
    if img.shape[-1] == 1:
        # Convert the image to a NumPy array
        img_array = img.numpy()
        # Convert grayscale to RGB
        rgb_image = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
        img = tf.convert_to_tensor(rgb_image)

    # Resize the image
    img = tf.image.resize(img, size = [img_shape, img_shape])
    # Rescale the image (if required)
    img = img/255. #(Only if your model expects pixel values in the range [0, 1])
    return img