import turicreate as tc
from os.path import basename

# Load the audio data and meta data.
data = tc.load_audio('audio-normalized/')
meta_data = tc.SFrame.read_csv('metadata.csv')

# Join the audio data and the meta data.
data['track-id'] = data['path'].apply(lambda p: basename(p))
data = data.join(meta_data)

# Drop all records which are not part of the ESC-10.
data = data.filter_by('TRUE', 'data')

# Make a train-test split, just use the first fold as our test set.
test_set = data.filter_by(5, 'fold')
train_set = data.filter_by(5, 'fold', exclude=True)

# Create the model.
model = tc.sound_classifier.create(train_set, target='floor', feature='audio', max_iterations = 400, batch_size=64)


# Generate an SArray of predictions from the test set.
predictions = model.predict(test_set)

# Evaluate the model and print the results
metrics = model.evaluate(test_set)
print(metrics)

# Save the model for later use in Turi Create
model.save('my_sound_classifier.model')

# Export for use in Core ML
model.export_coreml('my_sound_classifier.mlmodel')
