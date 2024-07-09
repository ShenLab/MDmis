import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import gridspec
import numpy as np

pd.set_option("display.max_columns", 100)

models_dir = "/home/az2798/IDR_cons/intermediate_models/"
results_dir = "/home/az2798/IDR_cons/results/"

training_loss_file = "_final_trainingloss.npy"
validation_loss_file = "_final_validationloss.npy"

training_loss = np.load(f'{models_dir}{training_loss_file}')
validation_loss = np.load(f'{models_dir}{validation_loss_file}')

epochs = np.arange(1, len(training_loss) + 1)
loss_df = pd.DataFrame({
    'epoch': epochs,
    'training_loss': training_loss,
    'validation_loss': validation_loss
})

# Plotting the loss over epochs
plt.figure(figsize=(10, 6))
sns.lineplot(x='epoch', y='training_loss', data=loss_df, label='Training Loss')
sns.lineplot(x='epoch', y='validation_loss', data=loss_df, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss over Epochs')
plt.legend()
plt.grid(True)
plt.savefig(f'{results_dir}train_val_loss_updated.png', dpi = 300, bbox_inches = "tight")