from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from matplotlib import pyplot as plt
import numpy as np

# Step 1: Make Predictions with the Model
y_pred = model.predict(np.array(train_x))
y_pred_classes = np.argmax(y_pred, axis=1)

# Step 2: Convert True Labels to Integer Classes
y_true = np.argmax(np.array(train_y), axis=1)

# Step 3: Generate the Confusion Matrix
cm = confusion_matrix(y_true, y_pred_classes)

# Step 4: Plot the Confusion Matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
disp.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.xlabel('Predicted Class')
plt.ylabel('True Class')
plt.show()
