import DeepTrap
#import comet_ml
experiment = Experiment(api_key="ypQZhYfs3nSyKzOfz13iuJpj2", project_name='deeplidar', log_code=True)

#Read config file
config = DeepTrap.utils.read_config()

#Load image data
data = DeepTrap.utils.read_image_data()

#load annotations
annotations = DeepTrap.utils.read_annotation_data()

#Create keras generator
generator = DeepTrap.Generator(data,annotations,config)

#Create callbacks
callbacks = DeepTrap.callback.create(config)

#Load Model
model = DeepTrap.model.create()

#Train Model
history = model.fit_generator(
    generator=generator,
    steps_per_epoch=generator.size()/config["batch_size"],
    epochs=config["batch_size"],
    verbose=2,
    callbacks=callbacks
)