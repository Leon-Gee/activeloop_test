import hub

dataset_path = 'hub://activeloop/mnist-train'

#lazy ftw
ds = hub.load(dataset_path)

# Indexing
img = ds.images[0].numpy()
label = ds.labels[0].numpy(aslist=True)

text_labels = ds.labels[0].data()['text']
print(text_labels)


# Slicing

imgs = ds.images[0:100].numpy()


labels = ds.labels[0:100].numpy(aslist=True)

