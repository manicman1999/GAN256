var model;
tf.loadModel('https://matchue.ca/p/earthgan/model/model.json').then((x) => {
    model = x;
    console.log(model)
    perform()
});