"""
Export Models Script
Run this AFTER running the finalNB.ipynb notebook to export the trained models.

Usage: 
1. Run all cells in finalNB.ipynb first
2. Then run this script in the same kernel/environment:
   %run export_models.py
"""

import pickle

print("="*60)
print("EXPORTING MODELS FOR DEPLOYMENT")
print("="*60)

try:
    # Save Neural Network Model
    model.save('model_nn.h5')
    print("✓ Neural Network saved as 'model_nn.h5'")
except Exception as e:
    print(f"✗ Neural Network export failed: {e}")

try:
    # Save Gradient Boosting Model
    with open('model_gb.pkl', 'wb') as f:
        pickle.dump(gb_model, f)
    print("✓ Gradient Boosting saved as 'model_gb.pkl'")
except Exception as e:
    print(f"✗ Gradient Boosting export failed: {e}")

try:
    # Save Random Forest Model
    with open('model_rf.pkl', 'wb') as f:
        pickle.dump(rf_model, f)
    print("✓ Random Forest saved as 'model_rf.pkl'")
except Exception as e:
    print(f"✗ Random Forest export failed: {e}")

try:
    # Save the StandardScaler
    with open('scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    print("✓ Scaler saved as 'scaler.pkl'")
except Exception as e:
    print(f"✗ Scaler export failed: {e}")

print("\n" + "="*60)
print("Export complete! Files ready for deployment:")
print("  - model_nn.h5")
print("  - model_gb.pkl")
print("  - model_rf.pkl") 
print("  - scaler.pkl")
print("="*60)
