import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as mp
import pickle
shap.initjs()


# ============================================================================
# Load data
# ============================================================================
data = pd.read_excel(r"data/final_dataset.xlsx", sheet_name='Sheet_R')
feature_names = data.columns
print(feature_names)
data = np.array(data)
print(data.shape)
# Split features and target variable for train/test dataset
x = data[:, :-1]
y = data[:, -1]
print(x.shape)
print(y.shape)  # Output data
print(x[0], y[0])
# Split dataset by target metal: 808 1046 899 943
Cr_x = x[:808, :]
Cr_y = y[:808]
Pb_x = x[808:1854, :]
Pb_y = y[808:1854]
Cd_x = x[1854:2753, :]
Cd_y = y[1854:2753]
As_x = x[2753:3696, :]
As_y = y[2753:3696]
X_display = pd.read_excel(r"data/final_dataset.xlsx", sheet_name='Sheet_TgMetuncoded')
X_display = np.array(X_display)
print(X_display.shape)
Cr_X_display = X_display[:808, :]
Pb_X_display = X_display[808:1854, :]
Cd_X_display = X_display[1854:2753, :]
As_X_display = X_display[2753:3696, :]
print(Pb_X_display[:, 0])
print(Pb_x[:, 0])
print(Cr_x.shape, Cr_y.shape, Cr_x[0], Cr_x[-1], Cr_y[0], Cr_y[-1])
print(Pb_x.shape, Pb_y.shape, Pb_x[0], Pb_x[-1], Pb_y[0], Pb_y[-1])
print(Cd_x.shape, Cd_y.shape, Cd_x[0], Cd_x[-1], Cd_y[0], Cd_y[-1])
print(As_x.shape, As_y.shape, As_x[0], As_x[-1], As_y[0], As_y[-1])

# ============================================================================
# Load model
# ============================================================================
with open('pkl/MOFsCrPbCdAs_XGB_R.pkl', 'rb') as f:
    model = pickle.load(f)  # Load model
    print('load success!')

# ============================================================================
# ============================================================================
# Global explanation
# ============================================================================
# ============================================================================

# ----------------------------------------------------------------------------
# SHAP-based feature importance
# ----------------------------------------------------------------------------
time_R = pd.read_excel(r"data/final_dataset.xlsx", sheet_name='Sheet_R_shap')
feature_names = time_R.columns
print(feature_names)
time_R = np.array(time_R)
print(time_R.shape)
time_R_x = time_R[:, :-1]
time_R_y = time_R[:, -1]
print(time_R_x.shape)
print(time_R_y.shape)
config = {
    "font.family": 'Times New Roman',
    "font.size": 8,
    "mathtext.fontset": 'stix',
}
mp.rcParams.update(config)
mp.rcParams['xtick.direction'] = 'in'
mp.rcParams['ytick.direction'] = 'in'
index_order = range(len(time_R_x))
print(index_order)
explainer = shap.Explainer(model, time_R_x, feature_names=feature_names)
shap_values_time_R = explainer(time_R_x)
shap.plots.heatmap(shap_values_time_R, max_display=14)
mp.gcf().set_size_inches(5, 4)
mp.colorbar(location='right')
mp.title('(b)MR', fontsize=14)
mp.show()

# ----------------------------------------------------------------------------
# SHAP based feature interaction importance
# ----------------------------------------------------------------------------
explainer = shap.TreeExplainer(model)
shap_interaction_valuestime = explainer.shap_interaction_values(x)
feature_namestime = feature_names[:-1]
fig = mp.figure(figsize=(7, 3))
grid = mp.GridSpec(1, 1, hspace=0.4, wspace=0.4)
ax1 = fig.add_subplot(grid[0, 0])
shap.summary_plot(shap_interaction_valuestime, x, max_display=20, feature_names=feature_namestime, show=False, plot_type="compact_dot")
ax1.set_title('(b)MR', fontsize=14)
mp.subplots_adjust(top=0.95, bottom=0.15, left=0.2, right=0.6)
mp.show()

# ============================================================================
# ============================================================================
# Local explanation (based on target heavy metal partitioning)
# ============================================================================
# ============================================================================

# ----------------------------------------------------------------------------
# Feature importance of different target heavy metals
# ----------------------------------------------------------------------------
explainer = shap.TreeExplainer(model)
shap_values1 = explainer.shap_values(Cr_x)
shap_values2 = explainer.shap_values(Pb_x)
shap_values3 = explainer.shap_values(Cd_x)
shap_values4 = explainer.shap_values(As_x)
config = {
    "font.family": 'Times New Roman',
    "font.size": 8,
    "mathtext.fontset": 'stix',
}
mp.rcParams.update(config)
mp.rcParams['xtick.direction'] = 'in'
mp.rcParams['ytick.direction'] = 'in'
fig = mp.figure(figsize=(7, 6))
grid = mp.GridSpec(2, 2, hspace=0.4, wspace=0.4)
ax1 = fig.add_subplot(grid[0, 0])
shap.summary_plot(shap_values1, Cr_x, feature_names=feature_names, show=False)
ax1.set_title('(e)Cr', fontsize=14)
ax2 = fig.add_subplot(grid[0, 1])
shap.summary_plot(shap_values2, Pb_x, feature_names=feature_names, show=False)
ax2.set_title('(f)Pb', fontsize=14)
ax3 = fig.add_subplot(grid[1, 0])
shap.summary_plot(shap_values3, Cd_x, feature_names=feature_names, show=False)
ax3.set_title('(g)Cd', fontsize=14)
ax4 = fig.add_subplot(grid[1, 1])
shap.summary_plot(shap_values4, As_x, feature_names=feature_names, show=False)
ax4.set_title('(h)As', fontsize=14)
mp.tight_layout()
mp.show()

# ----------------------------------------------------------------------------
# Feature interaction importance of different target heavy metals
# ----------------------------------------------------------------------------
explainer = shap.TreeExplainer(model)
shap_interaction_values5 = explainer.shap_interaction_values(Cr_x)
shap_interaction_values6 = explainer.shap_interaction_values(Pb_x)
shap_interaction_values7 = explainer.shap_interaction_values(Cd_x)
shap_interaction_values8 = explainer.shap_interaction_values(As_x)
config = {
    "font.family": 'Times New Roman',
    "font.size": 8,
    "mathtext.fontset": 'stix',
}
mp.rcParams.update(config)
mp.rcParams['xtick.direction'] = 'in'
mp.rcParams['ytick.direction'] = 'in'
feature_names5 = feature_names[:-1]
fig = mp.figure(figsize=(7, 7))
grid = mp.GridSpec(2, 2, hspace=0.3, wspace=0.5)
ax1 = fig.add_subplot(grid[0, 0])
shap.summary_plot(shap_interaction_values5, Cr_x, max_display=15, feature_names=feature_names5, show=False, plot_type="compact_dot")
ax1.set_title('(e)Cr', fontsize=14)
ax2 = fig.add_subplot(grid[0, 1])
shap.summary_plot(shap_interaction_values6, Pb_x, max_display=15, feature_names=feature_names5, show=False, plot_type="compact_dot")
ax2.set_title('(f)Pb', fontsize=14)
ax3 = fig.add_subplot(grid[1, 0])
shap.summary_plot(shap_interaction_values7, Cd_x, max_display=15, feature_names=feature_names5, show=False, plot_type="compact_dot")
ax3.set_title('(g)Cd', fontsize=14)
ax4 = fig.add_subplot(grid[1, 1])
shap.summary_plot(shap_interaction_values8, As_x, max_display=15, feature_names=feature_names5, show=False, plot_type="compact_dot")
ax4.set_title('(h)As', fontsize=14)
mp.tight_layout()
mp.show()

# ----------------------------------------------------------------------------
# Interaction feature dependence of different target heavy metals
# ----------------------------------------------------------------------------
config = {
    "font.family": 'Times New Roman',
    "font.size": 8,
    "mathtext.fontset": 'stix',
}
mp.rcParams.update(config)
mp.rcParams['xtick.direction'] = 'in'
mp.rcParams['ytick.direction'] = 'in'
fig = mp.figure(figsize=(7, 4.6))
grid = mp.GridSpec(2, 2, hspace=0.5, wspace=0.35)
ax1 = fig.add_subplot(grid[0, 0])
shap.dependence_plot("MT", shap_values1, Cr_x, interaction_index='AD', feature_names=feature_names, show=False, display_features=Cr_X_display, ax=ax1)
ax1.set_title('(e)Cr', fontsize=14)
ax1.tick_params(axis='x', labelsize=7)
ax2 = fig.add_subplot(grid[0, 1])
shap.dependence_plot("MT", shap_values2, Pb_x, interaction_index='AD', feature_names=feature_names, show=False, display_features=Pb_X_display, ax=ax2)
ax2.set_title('(f)Pb', fontsize=14)
ax2.tick_params(axis='x', labelsize=7)
ax3 = fig.add_subplot(grid[1, 0])
shap.dependence_plot("MT", shap_values3, Cd_x, interaction_index='AD', feature_names=feature_names, show=False, display_features=Cd_X_display, ax=ax3)
ax3.set_title('(g)Cd', fontsize=14)
ax3.tick_params(axis='x', labelsize=7)
ax4 = fig.add_subplot(grid[1, 1])
shap.dependence_plot("MT", shap_values4, As_x, interaction_index='AD', feature_names=feature_names, show=False, display_features=As_X_display, ax=ax4)
ax4.set_title('(h)As', fontsize=14)
ax4.tick_params(axis='x', labelsize=7)
mp.subplots_adjust(top=0.90, bottom=0.1, left=0.13, right=0.95)
mp.tight_layout()
mp.show()

# ============================================================================
# Partial dependence based on SHAP. Partial dependence of MOF structures
# ============================================================================
fig = mp.figure(figsize=(7, 5))
grid = mp.GridSpec(2, 3, hspace=0.5, wspace=0.5)
ax1 = fig.add_subplot(grid[0, 0])
shap.dependence_plot("SA", shap_values1, Cr_x, interaction_index=None, feature_names=feature_names, show=False, ax=ax1)
ax1.set_title('(m)Cr', fontsize=14)
coefficients = np.polyfit(Cr_x[:, 3], shap_values1[:, 3], 2)  # 2 indicates quadratic polynomial
polynomial = np.poly1d(coefficients)  # Calculate fitted curve y-values over continuous independent variable range
x_range = np.linspace(min(Cr_x[:, 3]), max(Cr_x[:, 3]), 100)  # Continuous independent variable range
fitted_line = polynomial(x_range)  # Create figure and add fitted line on subplot
ax1.plot(x_range, fitted_line, color='r', label='Fitted line')  # Draw fitted line
ax2 = fig.add_subplot(grid[0, 1])
shap.dependence_plot("PD", shap_values1, Cr_x, interaction_index=None, feature_names=feature_names, show=False, ax=ax2)
ax2.set_title('(n)Cr', fontsize=14)
coefficients = np.polyfit(Cr_x[:, 4], shap_values1[:, 4], 2)  # 2 indicates quadratic polynomial
polynomial = np.poly1d(coefficients)  # Calculate fitted curve y-values over continuous independent variable range
x_range = np.linspace(min(Cr_x[:, 4]), max(Cr_x[:, 4]), 100)  # Continuous independent variable range
fitted_line = polynomial(x_range)  # Create figure and add fitted line on subplot
ax2.plot(x_range, fitted_line, color='r', label='Fitted line')  # Draw fitted line
ax3 = fig.add_subplot(grid[0, 2])
shap.dependence_plot("PV", shap_values1, Cr_x, interaction_index=None, feature_names=feature_names, show=False, ax=ax3)
ax3.set_title('(o)Cr', fontsize=14)
coefficients = np.polyfit(Cr_x[:, 5], shap_values1[:, 5], 2)  # 2 indicates quadratic polynomial
polynomial = np.poly1d(coefficients)  # Calculate fitted curve y-values over continuous independent variable range
x_range = np.linspace(min(Cr_x[:, 5]), max(Cr_x[:, 5]), 100)  # Continuous independent variable range
fitted_line = polynomial(x_range)  # Create figure and add fitted line on subplot
ax3.plot(x_range, fitted_line, color='r', label='Fitted line')  # Draw fitted line
ax4 = fig.add_subplot(grid[1, 0])
shap.dependence_plot("SA", shap_values2, Pb_x, interaction_index=None, feature_names=feature_names, show=False, ax=ax4)
ax4.set_title('(p)Pb', fontsize=14)
coefficients = np.polyfit(Pb_x[:, 3], shap_values2[:, 3], 2)  # 2 indicates quadratic polynomial
polynomial = np.poly1d(coefficients)  # Calculate fitted curve y-values over continuous independent variable range
x_range = np.linspace(min(Pb_x[:, 3]), max(Pb_x[:, 3]), 100)  # Continuous independent variable range
fitted_line = polynomial(x_range)  # Create figure and add fitted line on subplot
ax4.plot(x_range, fitted_line, color='r', label='Fitted line')  # Draw fitted line
ax5 = fig.add_subplot(grid[1, 1])
shap.dependence_plot("PD", shap_values2, Pb_x, interaction_index=None, feature_names=feature_names, show=False, ax=ax5)
ax5.set_title('(q)Pb', fontsize=14)
coefficients = np.polyfit(Pb_x[:, 4], shap_values2[:, 4], 2)  # 2 indicates quadratic polynomial
polynomial = np.poly1d(coefficients)  # Calculate fitted curve y-values over continuous independent variable range
x_range = np.linspace(min(Pb_x[:, 4]), max(Pb_x[:, 4]), 100)  # Continuous independent variable range
fitted_line = polynomial(x_range)  # Create figure and add fitted line on subplot
ax5.plot(x_range, fitted_line, color='r', label='Fitted line')  # Draw fitted line
ax6 = fig.add_subplot(grid[1, 2])
shap.dependence_plot("PV", shap_values2, Pb_x, interaction_index=None, feature_names=feature_names, show=False, ax=ax6)
ax6.set_title('(r)Pb', fontsize=14)
coefficients = np.polyfit(Pb_x[:, 5], shap_values2[:, 5], 2)  # 2 indicates quadratic polynomial
polynomial = np.poly1d(coefficients)  # Calculate fitted curve y-values over continuous independent variable range
x_range = np.linspace(min(Pb_x[:, 5]), max(Pb_x[:, 5]), 100)  # Continuous independent variable range
fitted_line = polynomial(x_range)  # Create figure and add fitted line on subplot
ax6.plot(x_range, fitted_line, color='r', label='Fitted line')  # Draw fitted line
ax6.set_xlim(0, 2.5)
mp.subplots_adjust(top=0.94, bottom=0.1, left=0.14, right=0.95)
mp.tight_layout()
mp.show()

# ============================================================================
# Partial dependence based on SHAP. Partial dependence of MOF structures
# ============================================================================
fig = mp.figure(figsize=(7, 5))
grid = mp.GridSpec(2, 3, hspace=0.5, wspace=0.5)
ax1 = fig.add_subplot(grid[0, 0])
shap.dependence_plot("SA", shap_values3, Cd_x, interaction_index=None, feature_names=feature_names, show=False, ax=ax1)
ax1.set_title('(s)Cd', fontsize=14)
coefficients = np.polyfit(Cd_x[:, 3], shap_values3[:, 3], 2)  # 2 indicates quadratic polynomial
polynomial = np.poly1d(coefficients)  # Calculate fitted curve y-values over continuous independent variable range
x_range = np.linspace(min(Cd_x[:, 3]), max(Cd_x[:, 3]), 100)  # Continuous independent variable range
fitted_line = polynomial(x_range)  # Create figure and add fitted line on subplot
ax1.plot(x_range, fitted_line, color='r', label='Fitted line')  # Draw fitted line
ax2 = fig.add_subplot(grid[0, 1])
shap.dependence_plot("PD", shap_values3, Cd_x, interaction_index=None, feature_names=feature_names, show=False, ax=ax2)
ax2.set_title('(t)Cd', fontsize=14)
coefficients = np.polyfit(Cd_x[:, 4], shap_values3[:, 4], 2)  # 2 indicates quadratic polynomial
polynomial = np.poly1d(coefficients)  # Calculate fitted curve y-values over continuous independent variable range
x_range = np.linspace(min(Cd_x[:, 4]), max(Cd_x[:, 4]), 100)  # Continuous independent variable range
fitted_line = polynomial(x_range)  # Create figure and add fitted line on subplot
ax2.plot(x_range, fitted_line, color='r', label='Fitted line')  # Draw fitted line
ax3 = fig.add_subplot(grid[0, 2])
shap.dependence_plot("PV", shap_values3, Cd_x, interaction_index=None, feature_names=feature_names, show=False, ax=ax3)
ax3.set_title('(u)Cd', fontsize=14)
coefficients = np.polyfit(Cd_x[:, 5], shap_values3[:, 5], 2)  # 2 indicates quadratic polynomial
polynomial = np.poly1d(coefficients)  # Calculate fitted curve y-values over continuous independent variable range
x_range = np.linspace(min(Cd_x[:, 5]), max(Cd_x[:, 5]), 100)  # Continuous independent variable range
fitted_line = polynomial(x_range)  # Create figure and add fitted line on subplot
ax3.plot(x_range, fitted_line, color='r', label='Fitted line')  # Draw fitted line
ax4 = fig.add_subplot(grid[1, 0])
shap.dependence_plot("SA", shap_values4, As_x, interaction_index=None, feature_names=feature_names, show=False, ax=ax4)
ax4.set_title('(v)As', fontsize=14)
coefficients = np.polyfit(As_x[:, 3], shap_values4[:, 3], 2)  # 2 indicates quadratic polynomial
polynomial = np.poly1d(coefficients)  # Calculate fitted curve y-values over continuous independent variable range
x_range = np.linspace(min(As_x[:, 3]), max(As_x[:, 3]), 100)  # Continuous independent variable range
fitted_line = polynomial(x_range)  # Create figure and add fitted line on subplot
ax4.plot(x_range, fitted_line, color='r', label='Fitted line')  # Draw fitted line
ax5 = fig.add_subplot(grid[1, 1])
shap.dependence_plot("PD", shap_values4, As_x, interaction_index=None, feature_names=feature_names, show=False, ax=ax5)
ax5.set_title('(w)As', fontsize=14)
coefficients = np.polyfit(As_x[:, 4], shap_values4[:, 4], 2)  # 2 indicates quadratic polynomial
polynomial = np.poly1d(coefficients)  # Calculate fitted curve y-values over continuous independent variable range
x_range = np.linspace(min(As_x[:, 4]), max(As_x[:, 4]), 100)  # Continuous independent variable range
fitted_line = polynomial(x_range)  # Create figure and add fitted line on subplot
ax5.plot(x_range, fitted_line, color='r', label='Fitted line')  # Draw fitted line
ax6 = fig.add_subplot(grid[1, 2])
shap.dependence_plot("PV", shap_values4, As_x, interaction_index=None, feature_names=feature_names, show=False, ax=ax6)
ax6.set_title('(x)As', fontsize=14)
coefficients = np.polyfit(As_x[:, 5], shap_values4[:, 5], 2)  # 2 indicates quadratic polynomial
polynomial = np.poly1d(coefficients)  # Calculate fitted curve y-values over continuous independent variable range
x_range = np.linspace(min(As_x[:, 5]), max(As_x[:, 5]), 100)  # Continuous independent variable range
fitted_line = polynomial(x_range)  # Create figure and add fitted line on subplot
ax6.plot(x_range, fitted_line, color='r', label='Fitted line')  # Draw fitted line
mp.subplots_adjust(top=0.94, bottom=0.1, left=0.14, right=0.95)
mp.tight_layout()
mp.show()

# ============================================================================
# Partial dependence based on SHAP. Partial dependence of experimental conditions
# ============================================================================
config = {
    "font.family": 'Times New Roman',
    "font.size": 8,
    "mathtext.fontset": 'stix',
}
mp.rcParams.update(config)
mp.rcParams['xtick.direction'] = 'in'
mp.rcParams['ytick.direction'] = 'in'
fig = mp.figure(figsize=(7, 5))
grid = mp.GridSpec(2, 3, hspace=0.5, wspace=0.5)
ax1 = fig.add_subplot(grid[0, 0])
shap.dependence_plot("pH", shap_values1, Cr_x, interaction_index=None, feature_names=feature_names, show=False, ax=ax1)
ax1.set_title('(m)Cr', fontsize=14)
coefficients = np.polyfit(Cr_x[:, 11], shap_values1[:, 11], 2)  # 2 indicates quadratic polynomial
polynomial = np.poly1d(coefficients)  # Calculate fitted curve y-values over continuous independent variable range
x_range = np.linspace(min(Cr_x[:, 11]), max(Cr_x[:, 11]), 100)  # Continuous independent variable range
fitted_line = polynomial(x_range)  # Create figure and add fitted line on subplot
ax1.plot(x_range, fitted_line, color='r', label='Fitted line')  # Draw fitted line
ax2 = fig.add_subplot(grid[0, 1])
shap.dependence_plot("Tim", shap_values1, Cr_x, interaction_index=None, feature_names=feature_names, show=False, ax=ax2)
ax2.set_title('(n)Cr', fontsize=14)
coefficients = np.polyfit(Cr_x[:, 10], shap_values1[:, 10], 2)  # 2 indicates quadratic polynomial
polynomial = np.poly1d(coefficients)  # Calculate fitted curve y-values over continuous independent variable range
x_range = np.linspace(min(Cr_x[:, 10]), max(Cr_x[:, 10]), 100)  # Continuous independent variable range
fitted_line = polynomial(x_range)  # Create figure and add fitted line on subplot
ax2.plot(x_range, fitted_line, color='r', label='Fitted line')  # Draw fitted line
ax3 = fig.add_subplot(grid[0, 2])
shap.dependence_plot("Tem", shap_values1, Cr_x, interaction_index=None, feature_names=feature_names, show=False, ax=ax3)
ax3.set_title('(o)Cr', fontsize=14)
coefficients = np.polyfit(Cr_x[:, 12], shap_values1[:, 12], 2)  # 2 indicates quadratic polynomial
polynomial = np.poly1d(coefficients)  # Calculate fitted curve y-values over continuous independent variable range
x_range = np.linspace(min(Cr_x[:, 12]), max(Cr_x[:, 12]), 100)  # Continuous independent variable range
fitted_line = polynomial(x_range)  # Create figure and add fitted line on subplot
ax3.plot(x_range, fitted_line, color='r', label='Fitted line')  # Draw fitted line
ax4 = fig.add_subplot(grid[1, 0])
shap.dependence_plot("pH", shap_values2, Pb_x, interaction_index=None, feature_names=feature_names, show=False, ax=ax4)
ax4.set_title('(p)Pb', fontsize=14)
coefficients = np.polyfit(Pb_x[:, 11], shap_values2[:, 11], 2)  # 2 indicates quadratic polynomial
polynomial = np.poly1d(coefficients)  # Calculate fitted curve y-values over continuous independent variable range
x_range = np.linspace(min(Pb_x[:, 11]), max(Pb_x[:, 11]), 100)  # Continuous independent variable range
fitted_line = polynomial(x_range)  # Create figure and add fitted line on subplot
ax4.plot(x_range, fitted_line, color='r', label='Fitted line')  # Draw fitted line
ax5 = fig.add_subplot(grid[1, 1])
shap.dependence_plot("Tim", shap_values2, Pb_x, interaction_index=None, feature_names=feature_names, show=False, ax=ax5)
ax5.set_title('(q)Pb', fontsize=14)
coefficients = np.polyfit(Pb_x[:, 10], shap_values2[:, 10], 2)  # 2 indicates quadratic polynomial
polynomial = np.poly1d(coefficients)  # Calculate fitted curve y-values over continuous independent variable range
x_range = np.linspace(min(Pb_x[:, 10]), max(Pb_x[:, 10]), 100)  # Continuous independent variable range
fitted_line = polynomial(x_range)  # Create figure and add fitted line on subplot
ax5.plot(x_range, fitted_line, color='r', label='Fitted line')  # Draw fitted line
ax6 = fig.add_subplot(grid[1, 2])
shap.dependence_plot("Tem", shap_values2, Pb_x, interaction_index=None, feature_names=feature_names, show=False, ax=ax6)
ax6.set_title('(r)Pb', fontsize=14)
coefficients = np.polyfit(Pb_x[:, 12], shap_values2[:, 12], 2)  # 2 indicates quadratic polynomial
polynomial = np.poly1d(coefficients)  # Calculate fitted curve y-values over continuous independent variable range
x_range = np.linspace(min(Pb_x[:, 12]), max(Pb_x[:, 12]), 100)  # Continuous independent variable range
fitted_line = polynomial(x_range)  # Create figure and add fitted line on subplot
ax6.plot(x_range, fitted_line, color='r', label='Fitted line')  # Draw fitted line
mp.subplots_adjust(top=0.94, bottom=0.1, left=0.14, right=0.95)
mp.tight_layout()
mp.show()

# ============================================================================
# Partial dependence based on SHAP. Partial dependence of experimental conditions
# ============================================================================
fig = mp.figure(figsize=(7, 5))
grid = mp.GridSpec(2, 3, hspace=0.5, wspace=0.5)
ax1 = fig.add_subplot(grid[0, 0])
shap.dependence_plot("pH", shap_values3, Cd_x, interaction_index=None, feature_names=feature_names, show=False, ax=ax1)
ax1.set_title('(s)Cd', fontsize=14)
coefficients = np.polyfit(Cd_x[:, 11], shap_values3[:, 11], 2)  # 2 indicates quadratic polynomial
polynomial = np.poly1d(coefficients)  # Calculate fitted curve y-values over continuous independent variable range
x_range = np.linspace(min(Cd_x[:, 11]), max(Cd_x[:, 11]), 100)  # Continuous independent variable range
fitted_line = polynomial(x_range)  # Create figure and add fitted line on subplot
ax1.plot(x_range, fitted_line, color='r', label='Fitted line')  # Draw fitted line
ax2 = fig.add_subplot(grid[0, 1])
shap.dependence_plot("Tim", shap_values3, Cd_x, interaction_index=None, feature_names=feature_names, show=False, ax=ax2)
ax2.set_title('(t)Cd', fontsize=14)
coefficients = np.polyfit(Cd_x[:, 10], shap_values3[:, 10], 2)  # 2 indicates quadratic polynomial
polynomial = np.poly1d(coefficients)  # Calculate fitted curve y-values over continuous independent variable range
x_range = np.linspace(min(Cd_x[:, 10]), max(Cd_x[:, 10]), 100)  # Continuous independent variable range
fitted_line = polynomial(x_range)  # Create figure and add fitted line on subplot
ax2.plot(x_range, fitted_line, color='r', label='Fitted line')  # Draw fitted line
ax3 = fig.add_subplot(grid[0, 2])
shap.dependence_plot("Tem", shap_values3, Cd_x, interaction_index=None, feature_names=feature_names, show=False, ax=ax3)
ax3.set_title('(u)Cd', fontsize=14)
coefficients = np.polyfit(Cd_x[:, 12], shap_values3[:, 12], 2)  # 2 indicates quadratic polynomial
polynomial = np.poly1d(coefficients)  # Calculate fitted curve y-values over continuous independent variable range
x_range = np.linspace(min(Cd_x[:, 12]), max(Cd_x[:, 12]), 100)  # Continuous independent variable range
fitted_line = polynomial(x_range)  # Create figure and add fitted line on subplot
ax3.plot(x_range, fitted_line, color='r', label='Fitted line')  # Draw fitted line
ax4 = fig.add_subplot(grid[1, 0])
shap.dependence_plot("pH", shap_values4, As_x, interaction_index=None, feature_names=feature_names, show=False, ax=ax4)
ax4.set_title('(v)As', fontsize=14)
coefficients = np.polyfit(As_x[:, 11], shap_values4[:, 11], 2)  # 2 indicates quadratic polynomial
polynomial = np.poly1d(coefficients)  # Calculate fitted curve y-values over continuous independent variable range
x_range = np.linspace(min(As_x[:, 11]), max(As_x[:, 11]), 100)  # Continuous independent variable range
fitted_line = polynomial(x_range)  # Create figure and add fitted line on subplot
ax4.plot(x_range, fitted_line, color='r', label='Fitted line')  # Draw fitted line
ax5 = fig.add_subplot(grid[1, 1])
shap.dependence_plot("Tim", shap_values4, As_x, interaction_index=None, feature_names=feature_names, show=False, ax=ax5)
ax5.set_title('(w)As', fontsize=14)
coefficients = np.polyfit(As_x[:, 10], shap_values4[:, 10], 2)  # 2 indicates quadratic polynomial
polynomial = np.poly1d(coefficients)  # Calculate fitted curve y-values over continuous independent variable range
x_range = np.linspace(min(As_x[:, 10]), max(As_x[:, 10]), 100)  # Continuous independent variable range
fitted_line = polynomial(x_range)  # Create figure and add fitted line on subplot
ax5.plot(x_range, fitted_line, color='r', label='Fitted line')  # Draw fitted line
ax6 = fig.add_subplot(grid[1, 2])
shap.dependence_plot("Tem", shap_values4, As_x, interaction_index=None, feature_names=feature_names, show=False, ax=ax6)
ax6.set_title('(x)As', fontsize=14)
coefficients = np.polyfit(As_x[:, 12], shap_values4[:, 12], 2)  # 2 indicates quadratic polynomial
polynomial = np.poly1d(coefficients)  # Calculate fitted curve y-values over continuous independent variable range
x_range = np.linspace(min(As_x[:, 12]), max(As_x[:, 12]), 100)  # Continuous independent variable range
fitted_line = polynomial(x_range)  # Create figure and add fitted line on subplot
ax6.plot(x_range, fitted_line, color='r', label='Fitted line')  # Draw fitted line
mp.subplots_adjust(top=0.94, bottom=0.1, left=0.14, right=0.95)
mp.tight_layout()
mp.show()
