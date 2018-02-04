from cnn_predict import predict_unseen_data

contents = []
fp = open('./testdataset/1.txt', 'r')

content = fp.readlines()
contents.append(content) 

result = predict_unseen_data(contents, 1)
print(result)
