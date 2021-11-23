from augmentts.datasets import ETSDataset

ets = ETSDataset(ets_families={
    'A,A,N' : 100,
    'A,A,A' : 10
}, length=100)

ts_data, family = ets.load(return_family=True)
print(family)
print(ts_data.shape)