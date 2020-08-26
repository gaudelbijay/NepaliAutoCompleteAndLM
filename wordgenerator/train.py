def train(model, optimizer, criterion, epochs, train_X, train_Y):
    for e in range(epochs):
        train_loss = 0
        #         train_acc = 0

        #         test_loss = 0
        #         test_acc = 0

        for i in range(len(train_X)):
            x = train_X[i]
            y = train_Y[i]
            optimizer.zero_grad()
            prediction, hidden = model(x)
            #             print(type(prediction))
            loss = criterion(prediction, y)
            loss.backward()
            optimizer.step()

            train_loss += loss.data.item()
        #             train_acc += torch.eq(prediction.round(),y).sum().item()
        train_loss /= len(train_X)
        #         train_acc /= len(train_X)
        print("Epoch {} Train Loss: {}".format(e + 1, train_loss))

#         for i in range(len(test_X)):
#             x = test_X[i].to(device)
#             y = test_Y[i:i+1].to(device)
#             y = y.float()
#             prediction = model(x)
#             prediction = prediction.squeeze(1)
#             loss = criterion(prediction,y)

#             test_loss += loss.data.item()
#             test_acc += torch.eq(prediction.round(),y).sum().item()
#         test_loss /= len(test_X)
#         test_acc /= len(test_X)

#         if (e+1)%10 == 0:
#             torch.save(model, "/content/drive/My Drive/models/SA_LSTM/model/e_" + str(e+1)+".bin")
#         print("Epoch: {}  Training (loss,acc): ({},{})  Test (loss, acc):({},{})".format( e+1, train_loss, train_acc, test_loss, test_acc))
