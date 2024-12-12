
import torch
from torch import nn,optim 
from tqdm import tqdm
from Encoder import Transformer
from torch.optim.lr_scheduler import ReduceLROnPlateau
from Dataset_creation import DataLoaderCreator
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sn
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.metrics import roc_curve, f1_score, cohen_kappa_score
from sklearn.metrics import roc_auc_score
device = torch.device("mps")
# sequence_len=16 # sequence length of time series
# max_len=5000 # max time series sequence length 
# n_head = 2 # number of attention head
# n_layer = 1# number of encoder layer
# drop_prob = 0.1
# d_model = 200 # number of dimension ( for positional embedding)
# ffn_hidden = 128 # size of hidden layer before classification 
# feature = 36 # for univariate time series (1d), it must be adjusted for 1. 
# batch_size = 32

sequence_len=128 # sequence length of time series
max_len=3004 # max time series sequence length 
n_head = 2 # number of attention head
n_layer = 8# number of encoder layer
drop_prob = 0.00025174311379616955
d_model = 200 # number of dimension ( for positional embedding)
ffn_hidden = 1981 # size of hidden layer before classification 
feature = 36 # for univariate time series (1d), it must be adjusted for 1. 
batch_size = 32
lr = 3.6976227597273716e-05

model =  Transformer(d_model=d_model, n_head=n_head, max_len=max_len, 
                     seq_len=sequence_len, ffn_hidden=ffn_hidden, n_layers=n_layer, 
                     drop_prob=drop_prob, details=False,device=device).to(device=device)


# train_csv_path='//Users//macbookpro//Documents//TSAD//Datasets//HAI_Binary_class//train_feature_selected1.csv'
# test_csv_path='//Users//macbookpro//Documents//TSAD//Datasets//HAI_Binary_class//test_feature_selected1.csv'

train_csv_path='//Users//macbookpro//Documents//TSAD//KDDE_Data//train1.csv'
test_csv_path='//Users//macbookpro//Documents//TSAD//KDDE_Data//test1.csv'

loader_creator = DataLoaderCreator(train_csv_path , test_csv_path, window_size=sequence_len, batch_size=batch_size, test_size=0.15)

train_loader, val_loader, test_loader = loader_creator.load_data()


for x_batch , y_batch in train_loader:
    x_batch = x_batch.to(device)
    print(x_batch.shape)
    print(y_batch.squeeze(-1).shape)
    break

out = model(x_batch)


weight_decay = 0.001

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=3)

epochs = 100

early_stopping_counter = 0
early_stopping_threshold = 5
best_val_loss = float('inf') 

train_losses = []
train_accuracy = []
validation_losses = []
validation_accuracy = []

best_val_accuracy = 0.0  
best_model = None  


for epoch in tqdm(range(epochs)):
    model.train()
    train_correct = 0
    total_train_correct = 0
    train_loss_epoch = 0

    for batch in train_loader:
        x_batch, y_batch = batch

        optimizer.zero_grad()
        
        outputs = model(x_batch)
        y_batch = y_batch.squeeze(-1)
        
        train_loss = criterion(outputs, y_batch)
        
        train_loss_epoch += train_loss.item()
        
        train_loss.backward()
        optimizer.step()

        _, predicted = torch.max(outputs, dim=1)
        train_correct += torch.sum(predicted == y_batch).item()
        total_train_correct += y_batch.size(0) 
    
    train_accuracy_epoch = np.array([100 * train_correct / total_train_correct])
    train_losses_epoch = np.array([train_loss_epoch / len(train_loader)])

    train_accuracy.append(train_accuracy_epoch)
    train_losses.append(train_losses_epoch)

    # Validation
    model.eval()
    val_correct = 0
    total_val_correct = 0
    val_loss_epoch = 0

    with torch.no_grad():
        for batch in val_loader:
            x_batch, y_batch = batch
            y_batch = y_batch.squeeze(-1)

            outputs = model(x_batch)
            val_loss = criterion(outputs, y_batch)
            val_loss_epoch += val_loss.item()

            _, predicted = torch.max(outputs, dim=1)
            val_correct += torch.sum(predicted == y_batch).item()
            total_val_correct += y_batch.size(0)

    validation_accuracy_epoch = np.array([100 * val_correct / total_val_correct])
    validation_losses_epoch = np.array([val_loss_epoch / len(val_loader)])

    validation_accuracy.append(validation_accuracy_epoch)
    validation_losses.append(validation_losses_epoch)

    scheduler.step(validation_losses_epoch[-1])  # Adjust learning rate based on validation loss
    
    # Early stopping check
    if validation_losses_epoch > best_val_loss:
        early_stopping_counter += 1
        if early_stopping_counter >= early_stopping_threshold:
            print("Early stopping triggered!")
            break
    else:
        early_stopping_counter = 0
        best_val_loss = validation_losses_epoch

    print(f" Epoch {epoch + 1}/{epochs}, train loss: {train_losses_epoch[0]:.4f}, train acc: {train_accuracy_epoch[0]:.4f}, val loss: {validation_losses_epoch[0]:.4f}, val acc: {validation_accuracy_epoch[0]:.4f}")
    if validation_accuracy_epoch > best_val_accuracy:
        best_val_accuracy = validation_accuracy_epoch
        best_model = model.state_dict()
        print("Saved model changed to: ", validation_accuracy_epoch)
    else:
        print("best model is not changed...")
    #print("Best model is not changed")
        
torch.save(best_model, '//Users//macbookpro//Documents//TSAD//Relative_PE//best_model.pth')


plt.figure(figsize=(10, 5))
plt.plot([i for i in range(len(train_losses))], train_losses, label='train loss    ', marker='o')
plt.plot([i for i in range(len(validation_losses))], validation_losses, label='val loss      ', marker='^')
plt.title('XL attention training losses vs. validation losses')
plt.xlabel('Epochs')
plt.ylabel('Losses (%)')
plt.legend(loc='lower right', bbox_to_anchor=(0.98, 0.18))
plt.grid(True)        
plt.show()


plt.figure(figsize=(10, 5))
plt.plot([i for i in range(len(train_accuracy))], train_accuracy, label='train accuracy', marker='o')
plt.plot([i for i in range(len(validation_accuracy))], validation_accuracy, label='val accuracy', marker='^')
plt.title('XL attention training vs. validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracies (%)')
plt.legend(loc='upper right', bbox_to_anchor=(0.98, 0.89))
plt.grid(True)
plt.show()


fig, ax1 = plt.subplots(figsize=(6, 3))

# Plot train and validation accuracies on the primary y-axis
ax1.plot([i for i in range(len(train_accuracy))], train_accuracy, 'b', label='train accuracy', marker='o')
ax1.plot([i for i in range(len(validation_accuracy))], validation_accuracy, 'r', label='val accuracy', marker='^')
ax1.set_xlabel('Epochs', fontdict={'size': 14, 'family': 'Times New Roman'})
ax1.set_ylabel('Accuracies (%)', fontdict={'size': 14, 'family': 'Times New Roman'})
ax1.tick_params(axis='y', labelsize=14)
ax1.legend(loc='upper right', bbox_to_anchor=(0.98, 0.89), prop={'size': 14, 'family': 'Times New Roman'})

# Create a secondary y-axis for losses
ax2 = ax1.twinx()
ax2.plot([i for i in range(len(train_losses))], train_losses, 'g', label='train loss', marker='o')
ax2.plot([i for i in range(len(validation_losses))], validation_losses, 'm', label='val loss', marker='^')
ax2.set_ylabel('Losses (%)', fontdict={'size': 14, 'family': 'Times New Roman'})
ax2.tick_params(axis='y', labelsize=14)
ax2.legend(loc='lower right', bbox_to_anchor=(0.98, 0.18), prop={'size': 14, 'family': 'Times New Roman'})

plt.title('Global attention training and validation accuracy vs. loss', fontdict={'size': 14, 'family': 'Times New Roman'})


# 'weight': 'bold'


# Set tick labels font properties
ax1.tick_params(axis='x', labelsize=12)
ax2.tick_params(axis='x', labelsize=12)
for tick in ax1.get_xticklabels():
    tick.set_fontname('Times New Roman')
    #tick.set_fontweight('bold')
for tick in ax1.get_yticklabels():
    tick.set_fontname('Times New Roman')
    #tick.set_fontweight('bold')
for tick in ax2.get_yticklabels():
    tick.set_fontname('Times New Roman')
    #tick.set_fontweight('bold')

plt.grid(True)
plt.show()


test_accuracy = []
test_losses = []
predicted_values = []

model.eval()
#model.load_state_dict(torch.load('//Users//macbookpro//Documents//TSAD//Relative_PE//best_model.pth')) 
test_correct = 0
total_test_correct = 0
test_loss_epoch = 0

y_test = []
y_pred = []
with torch.no_grad():
    for batch in tqdm(test_loader):
        x_batch, y_batch = batch
      
        y_batch = y_batch.squeeze(-1)
        y_test.extend(y_batch.detach().cpu().numpy())

        outputs = model(x_batch)
        test_loss = criterion(outputs, y_batch)
        test_loss_epoch += test_loss.item()

        _, predicted = torch.max(outputs, dim=1)
        test_correct += torch.sum(predicted == y_batch).item()
        predicted_values.extend(outputs.detach().cpu().numpy())
        y_pred.extend(predicted.detach().cpu().numpy())
        total_test_correct += y_batch.size(0)
        
    test_accuracy_epoch = 100 * test_correct / total_test_correct
    test_losses_epoch = test_loss_epoch / len(test_loader)

    test_accuracy.append(test_accuracy_epoch)
    test_losses.append(test_losses_epoch)
    print(f"Test Accuracy: {test_accuracy_epoch:.2f}%, Test Loss: {test_losses_epoch:.4f}")

print(classification_report(y_test, y_pred))

# Generate confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Define class labels
class_labels = ['Normal', 'Abnormal']

# Create DataFrame with class labels
df_cm = pd.DataFrame(cm, index=class_labels, columns=class_labels)

plt.figure(figsize=(6, 4))  # Set the figure size

# Plot heatmap
ax = sn.heatmap(df_cm, annot=True, fmt='d', cmap="viridis", linewidths=1, annot_kws={"size": 14, "fontfamily": "Times New Roman"})

# Set labels for x and y axes
ax.set_xlabel('Predicted labels', fontdict={'size': 14, 'family': 'Times New Roman'})
ax.set_ylabel('True labels', fontdict={'size': 14, 'family': 'Times New Roman'})
plt.title("Global attention Confusion Matrix", fontdict={'size': 14, 'family': 'Times New Roman'})

# Set tick labels font properties
ax.tick_params(axis='both', which='major', labelsize=14)
for tick in ax.get_xticklabels():
    tick.set_fontname('Times New Roman')
    #tick.set_fontweight('bold')
for tick in ax.get_yticklabels():
    tick.set_fontname('Times New Roman')
    #tick.set_fontweight()

plt.show()

print("Accuracy : ", accuracy_score(y_test, y_pred))
print("Precision: ", precision_score(y_test, y_pred))
print("recall:    ", recall_score(y_test, y_pred))
print("F1-score:  ", f1_score(y_test, y_pred))
print("AUC:       ", roc_auc_score(y_test, y_pred))
print("Kappa:     ", cohen_kappa_score(y_test, y_pred))

# Generate ROC curve and AUC score
fpr, tpr, _ = roc_curve(y_test, y_pred)
auc = roc_auc_score(y_test, y_pred)

# Create ROC curve plot
plt.figure(figsize=(6, 4))  # Set the figure size
plt.plot(fpr, tpr, label="AUC=" + str(round(auc, 6)), linewidth=2)
plt.plot([0, 1], [0, 1], 'k--', label='No Skill', linewidth=2)
plt.ylabel('True Positive Rate', fontdict={'size': 14, 'family': 'Times New Roman'})
plt.xlabel('False Positive Rate', fontdict={'size': 14, 'family': 'Times New Roman'})
plt.title("Global attention AUC ROC Curve", fontdict={'size': 14, 'family': 'Times New Roman'})
plt.legend(loc=4, prop={'size': 14, 'family': 'Times New Roman'})

# Set tick labels font properties
plt.xticks(fontsize=14, fontname='Times New Roman')
plt.yticks(fontsize=14, fontname='Times New Roman')



