import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import warnings
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model

warnings.filterwarnings('ignore')

# Load the neural network model
try:
    model = load_model('diabetes_nn_model_enhanced.keras')
except Exception as e:
    print(f"Error loading model: {e}")
    exit(1)

# Define column names for Pima dataset
pima_columns = [
    'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
    'Insulin', 'BMI', 'Insulin_Pedigree', 'Age', 'Outcome'
]

# Load original Pima dataset to fit scaler
try:
    original_data = pd.read_csv('pima-indians-diabetes.data.csv', names=pima_columns)
    print("Dataset loaded successfully. Columns:", original_data.columns.tolist())
except FileNotFoundError:
    print("Error: pima-indians-diabetes.data.csv not found.")
    exit(1)
except Exception as e:
    print(f"Error loading dataset: {e}")
    exit(1)

# Convert all feature columns to numeric, coercing errors to NaN
base_features = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
                 'Insulin', 'BMI', 'Insulin_Pedigree', 'Age']
for col in base_features:
    original_data[col] = pd.to_numeric(original_data[col], errors='coerce')

# Check for NaN values and replace with column mean
for col in base_features:
    if original_data[col].isna().any():
        mean_value = original_data[col].mean()
        original_data[col].fillna(mean_value, inplace=True)
        print(f"Replaced NaN in {col} with mean: {mean_value}")

# Verify data types
print("Data types after conversion:\n", original_data[base_features].dtypes)

# Prepare data for scaler, including derived features
X_train_base = original_data[base_features]
# Scale base features first
scaler_base = StandardScaler()
X_train_base_scaled = scaler_base.fit_transform(X_train_base)
X_train_base_scaled = pd.DataFrame(X_train_base_scaled, columns=base_features)
# Compute derived features using scaled values
X_train_derived = pd.DataFrame({
    'Log_Insulin': np.log1p(X_train_base['Insulin']),
    'Glucose_BMI': X_train_base_scaled['Glucose'] * X_train_base_scaled['BMI']
})
X_train_full = pd.concat([X_train_base, X_train_derived], axis=1)

# Model expects features in this order
model_features = ['Pregnancies', 'Age', 'BloodPressure', 'Insulin', 'Insulin_Pedigree', 
                 'SkinThickness', 'Log_Insulin', 'Glucose', 'Glucose_BMI', 'BMI']

# Fit scaler on full feature set
scaler = StandardScaler()
scaler.fit(X_train_full[model_features])
print("Scaler mean:", scaler.mean_)
print("Scaler scale:", scaler.scale_)

# Fuzzy Logic System Setup
risk = ctrl.Antecedent(np.arange(0, 1.01, 0.01), 'risk')
glucose = ctrl.Antecedent(np.arange(0, 401, 1), 'glucose')
bmi = ctrl.Antecedent(np.arange(0, 81, 0.5), 'bmi')
blood_pressure = ctrl.Antecedent(np.arange(50, 151, 1), 'blood_pressure')
age = ctrl.Antecedent(np.arange(20, 101, 1), 'age')
output = ctrl.Consequent(np.arange(0, 1.01, 0.01), 'output')

# Membership Functions
glucose['low'] = fuzz.trapmf(glucose.universe, [0, 0, 60, 80])
glucose['normal'] = fuzz.trapmf(glucose.universe, [70, 90, 150, 170])
glucose['high'] = fuzz.trapmf(glucose.universe, [160, 190, 400, 400])
bmi['low'] = fuzz.trapmf(bmi.universe, [0, 0, 18, 22])
bmi['normal'] = fuzz.trapmf(bmi.universe, [18, 23, 33, 38])
bmi['high'] = fuzz.trapmf(bmi.universe, [35, 40, 80, 80])
blood_pressure['low'] = fuzz.trapmf(blood_pressure.universe, [50, 50, 55, 65])
blood_pressure['normal'] = fuzz.trapmf(blood_pressure.universe, [60, 70, 90, 100])
blood_pressure['high'] = fuzz.trapmf(blood_pressure.universe, [95, 110, 150, 150])
age['young'] = fuzz.trapmf(age.universe, [20, 20, 25, 35])
age['middle'] = fuzz.trapmf(age.universe, [30, 35, 50, 60])
age['old'] = fuzz.trapmf(age.universe, [55, 65, 100, 100])
risk['low'] = fuzz.trapmf(risk.universe, [0, 0, 0.3, 0.4])
risk['medium'] = fuzz.trapmf(risk.universe, [0.35, 0.5, 0.65, 0.75])
risk['high'] = fuzz.trapmf(risk.universe, [0.7, 0.8, 1, 1])
output['low'] = fuzz.trapmf(output.universe, [0, 0, 0.3, 0.4])
output['medium'] = fuzz.trapmf(output.universe, [0.3, 0.45, 0.65, 0.8])
output['high'] = fuzz.trapmf(output.universe, [0.75, 0.85, 1, 1])

# Fuzzy Rules (all 48 rules)
rule1 = ctrl.Rule(risk['high'] & glucose['high'] & bmi['high'], output['high'])
rule2 = ctrl.Rule(risk['high'] & glucose['high'] & bmi['normal'], output['high'])
rule3 = ctrl.Rule(risk['medium'] & glucose['normal'] & bmi['normal'], output['low'])
rule4 = ctrl.Rule(risk['medium'] & glucose['high'] & bmi['high'], output['medium'])
rule5 = ctrl.Rule(risk['low'] & glucose['normal'] & bmi['normal'], output['low'])
rule6 = ctrl.Rule(risk['low'] & glucose['low'] & bmi['low'], output['low'])
rule7 = ctrl.Rule(risk['high'] & glucose['normal'] & bmi['normal'], output['medium'])
rule8 = ctrl.Rule(risk['medium'] & glucose['low'] & bmi['normal'], output['low'])
rule9 = ctrl.Rule(risk['low'] & glucose['high'] & bmi['high'], output['medium'])
rule10 = ctrl.Rule(risk['medium'] & glucose['high'] & bmi['normal'], output['high'])
rule11 = ctrl.Rule(risk['low'] & glucose['high'] & bmi['normal'], output['medium'])
rule12 = ctrl.Rule(risk['high'] & glucose['low'] & bmi['low'], output['medium'])
rule13 = ctrl.Rule(risk['low'] & glucose['low'] & bmi['high'], output['low'])
rule14 = ctrl.Rule(risk['medium'] & glucose['low'] & bmi['high'], output['medium'])
rule15 = ctrl.Rule(risk['high'] & glucose['normal'] & bmi['high'], output['high'])
rule16 = ctrl.Rule(risk['medium'] & glucose['normal'] & bmi['high'], output['medium'])
rule17 = ctrl.Rule(risk['low'] & glucose['normal'] & bmi['high'], output['low'])
rule18 = ctrl.Rule(risk['high'] & glucose['high'] & bmi['low'], output['high'])
rule19 = ctrl.Rule(risk['medium'] & glucose['normal'] & bmi['low'], output['low'])
rule20 = ctrl.Rule(risk['high'] & glucose['low'] & bmi['high'], output['high'])
rule21 = ctrl.Rule(risk['medium'] & glucose['high'] & bmi['low'], output['medium'])
rule22 = ctrl.Rule(risk['low'] & glucose['low'] & bmi['normal'], output['low'])
rule23 = ctrl.Rule(risk['high'] & glucose['normal'] & bmi['low'], output['medium'])
rule24 = ctrl.Rule(risk['medium'] & glucose['low'] & bmi['low'], output['low'])
rule25 = ctrl.Rule(risk['high'] & glucose['high'] & bmi['high'] & blood_pressure['high'], output['high'])
rule26 = ctrl.Rule(risk['medium'] & glucose['normal'] & bmi['normal'] & blood_pressure['low'], output['low'])
rule27 = ctrl.Rule(risk['low'] & glucose['low'] & bmi['low'] & blood_pressure['low'], output['low'])
rule28 = ctrl.Rule(risk['high'] & glucose['normal'] & bmi['normal'] & blood_pressure['low'], output['medium'])
rule29 = ctrl.Rule(risk['medium'] & glucose['low'] & bmi['high'] & blood_pressure['normal'], output['low'])
rule30 = ctrl.Rule(risk['low'] & glucose['high'] & bmi['low'] & blood_pressure['high'], output['medium'])
rule31 = ctrl.Rule(risk['high'] & glucose['high'] & bmi['high'] & age['old'], output['high'])
rule32 = ctrl.Rule(risk['medium'] & glucose['normal'] & bmi['normal'] & age['young'], output['low'])
rule33 = ctrl.Rule(risk['high'] & glucose['high'] & bmi['high'] & blood_pressure['normal'] & age['middle'], output['high'])
rule34 = ctrl.Rule(risk['medium'] & glucose['high'] & bmi['normal'] & blood_pressure['high'], output['high'])
rule35 = ctrl.Rule(risk['medium'] & glucose['high'] & bmi['high'] & age['old'], output['high'])
rule36 = ctrl.Rule(risk['high'] & glucose['normal'] & bmi['high'] & blood_pressure['high'], output['high'])
rule37 = ctrl.Rule(risk['low'] & glucose['low'] & bmi['normal'] & blood_pressure['normal'] & age['young'], output['low'])
rule38 = ctrl.Rule(risk['medium'] & glucose['normal'] & bmi['low'] & blood_pressure['low'], output['low'])
rule39 = ctrl.Rule(risk['high'] & glucose['high'] & bmi['high'] & blood_pressure['high'] & age['middle'], output['high'])
rule40 = ctrl.Rule(glucose['low'] & bmi['low'] & blood_pressure['low'] & age['young'], output['low'])
rule41 = ctrl.Rule(glucose['normal'] & bmi['normal'] & blood_pressure['normal'] & age['young'], output['low'])
rule42 = ctrl.Rule(glucose['normal'] & bmi['low'] & blood_pressure['low'] & age['middle'], output['low'])
rule43 = ctrl.Rule(glucose['high'] & bmi['high'] & blood_pressure['high'] & age['old'], output['high'])
rule44 = ctrl.Rule(glucose['normal'] & bmi['normal'] & blood_pressure['high'] & age['middle'], output['medium'])
rule45 = ctrl.Rule(glucose['low'] & bmi['normal'] & blood_pressure['normal'] & age['middle'], output['low'])
rule46 = ctrl.Rule(glucose['low'] & bmi['low'] & blood_pressure['low'] & age['young'], output['low'])
rule47 = ctrl.Rule(glucose['normal'] & bmi['normal'] & blood_pressure['normal'] & age['young'], output['low'])
rule48 = ctrl.Rule(glucose['normal'] & bmi['low'] & blood_pressure['low'] & age['middle'], output['low'])

fuzzy_system = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5, rule6, rule7, rule8, rule9, rule10,
                                  rule11, rule12, rule13, rule14, rule15, rule16, rule17, rule18, rule19,
                                  rule20, rule21, rule22, rule23, rule24, rule25, rule26, rule27, rule28,
                                  rule29, rule30, rule31, rule32, rule33, rule34, rule35, rule36, rule37,
                                  rule38, rule39, rule40, rule41, rule42, rule43, rule44, rule45, rule46,
                                  rule47, rule48])
fuzzy_simulation = ctrl.ControlSystemSimulation(fuzzy_system)

# Fuzzy prediction function
def predict_risk(nn_risk, glucose_val, bmi_val, blood_pressure_val, age_val):
    try:
        fuzzy_simulation.input['risk'] = nn_risk
        fuzzy_simulation.input['glucose'] = glucose_val
        fuzzy_simulation.input['bmi'] = bmi_val
        fuzzy_simulation.input['blood_pressure'] = blood_pressure_val
        fuzzy_simulation.input['age'] = age_val
        fuzzy_simulation.compute()
        risk_score = fuzzy_simulation.output.get('output', 0.5)
        if risk_score <= 0.35:
            risk_label = "Low Risk"
        elif risk_score <= 0.8:
            risk_label = "Medium Risk"
        else:
            risk_label = "High Risk"
        return risk_score, risk_label
    except Exception as e:
        print(f"Error in fuzzy computation: {e}")
        return 0.5, "Medium Risk"

# GUI Class with Updated Appearance Inspired by Image
class DiabetesPredictionGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Diabetes Risk Predictor")
        self.root.geometry("500x700")
        self.root.configure(bg="#d4f0e6")  # Light green background from image
        self.root.resizable(False, False)

        # Style configuration
        style = ttk.Style()
        style.configure("TLabel", font=("Roboto", 12), background="#d4f0e6")
        style.configure("TEntry", font=("Roboto", 12))
        style.configure("TButton", font=("Roboto", 12, "bold"))
        
        # Main container
        main_frame = ttk.Frame(root, padding=20, style="Main.TFrame")
        main_frame.pack(fill="both", expand=True)
        
        # Custom style for main frame
        style.configure("Main.TFrame", background="#d4f0e6")
        
        # Header
        header_frame = ttk.Frame(main_frame)
        header_frame.pack(fill="x", pady=(0, 20))
        ttk.Label(header_frame, text="Diabetes Risk Predictor", 
                 font=("Roboto", 22, "bold"), 
                 foreground="darkslategray",  # Black text like in image
                 background="#d4f0e6").pack()
        ttk.Label(header_frame, text="Enter patient data to predict diabetes risk", 
                 font=("Roboto", 10, "italic"), 
                 foreground="#1a3c5e",  # Dark blue from speech bubble
                 background="#d4f0e6").pack()

        # Input fields
        input_frame = ttk.LabelFrame(main_frame, text=" Patient Data ", 
                                   padding=15, style="Input.TLabelframe")
        input_frame.pack(fill="x", pady=10)
        
        # Style for LabelFrame
        style.configure("Input.TLabelframe", font=("Roboto", 12, "bold"), 
                       foreground="#1a3c5e")
        
        self.entries = {}
        labels = [
            ("Glucose (mg/dL, 0-400)", "Glucose"),
            ("BMI (0-80)", "BMI"),
            ("Blood Pressure (mmHg, 50-150)", "BloodPressure"),
            ("Age (years, 20-100)", "Age")
        ]
        
        for i, (label_text, key) in enumerate(labels):
            frame = ttk.Frame(input_frame)
            frame.pack(fill="x", pady=8)
            ttk.Label(frame, text=label_text, width=30, 
                     foreground="#1a3c5e").grid(row=i, column=0, padx=5, sticky="w")
            entry = ttk.Entry(frame, font=("Roboto", 12))
            entry.grid(row=i, column=1, padx=5, sticky="ew")
            self.entries[key] = entry
        input_frame.columnconfigure(1, weight=1)

        # Buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill="x", pady=20)
        
        def on_predict_enter(e):
            predict_btn.configure(style="PredictHover.TButton")
        def on_predict_leave(e):
            predict_btn.configure(style="Predict.TButton")
        
        style.configure("Predict.TButton", background="#6ec4b9",  # Teal from glucometer
                       foreground="teal", padding=8)
        style.map("Predict.TButton", background=[("active", "#5ab4a9")])  # Slightly darker teal
        style.configure("PredictHover.TButton", background="#5ab4a9", 
                       foreground="teal")
        
        predict_btn = ttk.Button(button_frame, text="Predict", 
                               command=self.predict, style="Predict.TButton")
        predict_btn.pack(side="left", padx=10)
        predict_btn.bind("<Enter>", on_predict_enter)
        predict_btn.bind("<Leave>", on_predict_leave)
        
        def on_clear_enter(e):
            clear_btn.configure(style="ClearHover.TButton")
        def on_clear_leave(e):
            clear_btn.configure(style="Clear.TButton")
        
        style.configure("Clear.TButton", background="#6ec4b9",  # Teal to match theme
                       foreground="teal", padding=8)
        style.map("Clear.TButton", background=[("active", "#5ab4a9")])
        style.configure("ClearHover.TButton", background="#5ab4a9", 
                       foreground="teal")
        
        clear_btn = ttk.Button(button_frame, text="Clear", 
                             command=self.clear_inputs, style="Clear.TButton")
        clear_btn.pack(side="left", padx=10)
        clear_btn.bind("<Enter>", on_clear_enter)
        clear_btn.bind("<Leave>", on_clear_leave)

        # Results
        results_frame = ttk.LabelFrame(main_frame, text=" Prediction Results ", 
                                     padding=15, style="Results.TLabelframe")
        results_frame.pack(fill="x", pady=10)
        
        style.configure("Results.TLabelframe", font=("Roboto", 12, "bold"), 
                       foreground="#1a3c5e")
        
        self.result_labels = {}
        result_fields = [
            ("Neural Network Probability:", "nn_prob", "#1a3c5e"),  # Dark blue
            ("Fuzzy Risk Score:", "fuzzy_score", "#1a3c5e"),
            ("Fuzzy Risk Label:", "fuzzy_label", "#4aa499"),  # Darker teal for contrast
            ("Ensemble Prediction:", "ensemble_pred", "#4aa499")
        ]
        
        for i, (label_text, key, color) in enumerate(result_fields):
            frame = ttk.Frame(results_frame)
            frame.pack(fill="x", pady=8)
            ttk.Label(frame, text=label_text, width=30, 
                     foreground="#1a3c5e").grid(row=i, column=0, padx=5, sticky="w")
            result_label = ttk.Label(frame, text="", 
                                   font=("Roboto", 12, "bold"), 
                                   foreground=color, background="#d4f0e6")
            result_label.grid(row=i, column=1, padx=5, sticky="ew")
            self.result_labels[key] = result_label
        results_frame.columnconfigure(1, weight=1)

    def clear_inputs(self):
        for entry in self.entries.values():
            entry.delete(0, tk.END)
        for label in self.result_labels.values():
            label.config(text="")

    def validate_inputs(self, inputs):
        ranges = {
            'Glucose': (0, 400),
            'BMI': (0, 80),
            'BloodPressure': (50, 150),
            'Age': (20, 100)
        }
        for key, value in inputs.items():
            try:
                val = float(value)
                min_val, max_val = ranges[key]
                if not (min_val <= val <= max_val):
                    return False, f"{key} must be between {min_val} and {max_val}"
            except ValueError:
                return False, f"{key} must be a valid number"
        return True, ""

    def predict(self):
        # Get inputs
        inputs = {key: entry.get() for key, entry in self.entries.items()}
        valid, error_msg = self.validate_inputs(inputs)
        
        if not valid:
            messagebox.showerror("Input Error", error_msg, parent=self.root)
            return

        # Prepare input for neural network
        glucose = float(inputs['Glucose'])
        bmi = float(inputs['BMI'])
        blood_pressure = float(inputs['BloodPressure'])
        age = float(inputs['Age'])

        # Create input array with all 8 base features
        input_base = np.array([[
            0,  # Pregnancies
            glucose,
            blood_pressure,
            0,  # SkinThickness
            0,  # Insulin
            bmi,
            0.5,  # Insulin_Pedigree (approximate mean)
            age
        ]])

        # Scale base features
        input_base_scaled = scaler_base.transform(input_base)
        input_base_scaled = pd.DataFrame(input_base_scaled, columns=base_features)

        # Compute derived features
        input_derived = np.array([[
            np.log1p(0),  # Log_Insulin (Insulin=0)
            input_base_scaled['Glucose'].iloc[0] * input_base_scaled['BMI'].iloc[0]  # Glucose_BMI
        ]])

        # Combine base and derived features
        input_full = np.concatenate([input_base, input_derived], axis=1)

        # Debug: Print raw input
        print("Raw input array:", input_full)

        # Scale inputs
        input_scaled = scaler.transform(input_full)

        # Debug: Print scaled input
        print("Scaled input array:", input_scaled)

        # Reorder features to match model input
        feature_indices = {
            'Pregnancies': 0,
            'Glucose': 1,
            'BloodPressure': 2,
            'SkinThickness': 3,
            'Insulin': 4,
            'BMI': 5,
            'Insulin_Pedigree': 6,
            'Age': 7,
            'Log_Insulin': 8,
            'Glucose_BMI': 9
        }
        input_model = np.zeros((1, 10))
        input_model[:, 0] = input_scaled[:, feature_indices['Pregnancies']]  # Pregnancies
        input_model[:, 1] = input_scaled[:, feature_indices['Age']]          # Age
        input_model[:, 2] = input_scaled[:, feature_indices['BloodPressure']] # BloodPressure
        input_model[:, 3] = input_scaled[:, feature_indices['Insulin']]      # Insulin
        input_model[:, 4] = input_scaled[:, feature_indices['Insulin_Pedigree']] # Insulin_Pedigree
        input_model[:, 5] = input_scaled[:, feature_indices['SkinThickness']] # SkinThickness
        input_model[:, 6] = input_scaled[:, feature_indices['Log_Insulin']]  # Log_Insulin
        input_model[:, 7] = input_scaled[:, feature_indices['Glucose']]      # Glucose
        input_model[:, 8] = input_scaled[:, feature_indices['Glucose_BMI']]  # Glucose_BMI
        input_model[:, 9] = input_scaled[:, feature_indices['BMI']]          # BMI

        # Debug: Print final input to model
        print("Final input to model:", input_model)

        # Neural network prediction
        try:
            nn_prob = model.predict(input_model, verbose=0)[0][0]
            print("Neural Network Probability:", nn_prob)
            if nn_prob < 1e-10 or nn_prob > 1 - 1e-10:
                print("Warning: Extreme nn_prob value detected. Check input scaling.")
        except Exception as e:
            messagebox.showerror("Prediction Error", f"Neural network prediction failed: {e}", parent=self.root)
            print(f"Prediction Error: {e}")
            return

        # Debug: Print nn_prob before GUI update
        print("nn_prob before GUI update:", nn_prob)

        # Fuzzy logic prediction
        glucose_val = np.clip(glucose, 0, 400)
        bmi_val = np.clip(bmi, 0, 80)
        blood_pressure_val = np.clip(blood_pressure, 50, 150)
        age_val = np.clip(age, 20, 100)
        
        fuzzy_score, fuzzy_label = predict_risk(nn_prob, glucose_val, bmi_val, 
                                               blood_pressure_val, age_val)
        print("Fuzzy Risk Score:", fuzzy_score, "Fuzzy Risk Label:", fuzzy_label)

        # Ensemble prediction
        combined_score = 0.8 * nn_prob + 0.2 * fuzzy_score
        ensemble_pred = "Diabetic" if combined_score > 0.6 else "Non-Diabetic"
        print("Combined Score:", combined_score, "Ensemble Prediction:", ensemble_pred)

        # Update result labels with error checking
        try:
            self.result_labels['nn_prob'].config(text=f"{nn_prob:.4f}")
            self.result_labels['fuzzy_score'].config(text=f"{fuzzy_score:.4f}")
            self.result_labels['fuzzy_label'].config(text=fuzzy_label)
            self.result_labels['ensemble_pred'].config(text=ensemble_pred)
            print("GUI updated successfully")
        except Exception as e:
            messagebox.showerror("GUI Update Error", f"Failed to update GUI: {e}", parent=self.root)
            print(f"GUI Update Error: {e}")

# Intro window with image
def show_intro():
    intro_root = tk.Tk()
    intro_root.title("Welcome")
    intro_root.geometry("500x700")  # Match main GUI size
    intro_root.configure(bg="#d4f0e6")
    intro_root.resizable(False, False)

    try:
        # Load the image (ensure the image file is named 'diabetes_intro.png' and in the same directory)
        from PIL import Image, ImageTk
        original_image = Image.open("diabetes_intro.jpeg")
        resized_image = original_image.resize((500, 700), Image.Resampling.LANCZOS)
        image = ImageTk.PhotoImage(resized_image)
        label = tk.Label(intro_root, image=image, bg="#d4f0e6")
        label.pack(pady=20)
        label.image = image  # Keep a reference to avoid garbage collection
    except Exception as e:
        print(f"Error loading intro image: {e}")
        # Fallback if image fails to load
        tk.Label(intro_root, text="Welcome to Diabetes Risk Predictor", 
                font=("Roboto", 16, "bold"), 
                fg="#000000", bg="#d4f0e6").pack(pady=20)

    # Close intro window after 3 seconds and open main GUI
    intro_root.after(3000, lambda: [intro_root.destroy(), start_main_gui()])
    intro_root.mainloop()

# Start the main GUI
def start_main_gui():
    root = tk.Tk()
    app = DiabetesPredictionGUI(root)
    root.mainloop()

# Main application
if __name__ == "__main__":
    show_intro()