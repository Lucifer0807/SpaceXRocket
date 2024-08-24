from flask import Flask, request, render_template, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load your model with the full path
model = joblib.load(r'model.pkl')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get form data
    data = request.form.to_dict()
    print("Form Data Received:", data)  # Debugging statement

    # Define feature names
    feature_names = [
        'FlightNumber', 'PayloadMass', 'Flights', 'Block', 'ReusedCount',
        'Orbit_ES-L1', 'Orbit_GEO', 'Orbit_GTO', 'Orbit_HEO', 'Orbit_ISS',
        'Orbit_LEO', 'Orbit_MEO', 'Orbit_PO', 'Orbit_SO', 'Orbit_SSO', 'Orbit_VLEO',
        'LaunchSite_CCAFS SLC 40', 'LaunchSite_KSC LC 39A', 'LaunchSite_VAFB SLC 4E',
        'LandingPad_5e9e3032383ecb267a34e7c7', 'LandingPad_5e9e3032383ecb554034e7c9',
        'LandingPad_5e9e3032383ecb761634e7cb', 'LandingPad_5e9e3032383ecb6bb234e7ca',
        'LandingPad_5e9e3033383ecbb9e534e7cc',
        'Serial_B0003', 'Serial_B0005', 'Serial_B0007', 'Serial_B1003', 'Serial_B1004',
        'Serial_B1005', 'Serial_B1006', 'Serial_B1007', 'Serial_B1008', 'Serial_B1010',
        'Serial_B1011', 'Serial_B1012', 'Serial_B1013', 'Serial_B1015', 'Serial_B1016',
        'Serial_B1017', 'Serial_B1018', 'Serial_B1019', 'Serial_B1020', 'Serial_B1021',
        'Serial_B1022', 'Serial_B1023', 'Serial_B1025', 'Serial_B1026', 'Serial_B1028',
        'Serial_B1029', 'Serial_B1030', 'Serial_B1031', 'Serial_B1032', 'Serial_B1034',
        'Serial_B1035', 'Serial_B1036', 'Serial_B1037', 'Serial_B1038', 'Serial_B1039',
        'Serial_B1040', 'Serial_B1041', 'Serial_B1042', 'Serial_B1043', 'Serial_B1044',
        'Serial_B1045', 'Serial_B1046', 'Serial_B1047', 'Serial_B1048', 'Serial_B1049',
        'Serial_B1050', 'Serial_B1051', 'Serial_B1054', 'Serial_B1056', 'Serial_B1058',
        'Serial_B1059', 'Serial_B1060', 'Serial_B1062',
        'GridFins_False', 'GridFins_True', 'Reused_False', 'Reused_True',
        'Legs_False', 'Legs_True'
    ]

    # Initialize feature values with default 0
    feature_values = dict.fromkeys(feature_names, 0)

    # Update feature values from form data
    for key in feature_names:
        if key in data:
                feature_values[key] = 1 if data[key] == 'on' else data[key]

    for key in data:
        feature_values[data[key]]= 1

    if(data['GridFins']=='True'):
        feature_values['GridFins_True']= 1
    else :
        feature_values['GridFins_False']= 1

    if (data['Reused'] == 'True'):
        feature_values['Reused_True'] = 1
    else:
        feature_values['Reused_False'] = 1

    if (data['Legs'] == 'True'):
        feature_values['Legs_True'] = 1
    else:
        feature_values['Legs_False'] = 1

    # Debugging feature_values dictionary
    print("Feature Values Dictionary:", feature_values)

    # Convert feature_values to list
    feature_values_list = [feature_values[name] for name in feature_names]
    print("Feature Values List:", feature_values_list)  # Debugging statement
    # Ensure feature values match the model's expectation
    feature_values_array = np.array(feature_values_list).reshape(1, -1)

    # Predict using the model
    prediction = model.predict(feature_values_array)[0]
    print("Prediction Result:", prediction)  # Debugging statement

    # Render result.html with the prediction
    return render_template('result.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
