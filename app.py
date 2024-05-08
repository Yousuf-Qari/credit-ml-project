import dash 
from dash import Dash, html, dcc
from dash.dependencies import Input, Output, State
import pickle5 as pickle
import pandas as pd

import sklearn
import pickle
import sys

# Check scikit-learn version
# required_sklearn_version = "0.XX.X"  # the version used during model training
# current_sklearn_version = sklearn.__version__

# if required_sklearn_version != current_sklearn_version:
#     print(f"Error: This model requires scikit-learn version {required_sklearn_version} but you have {current_sklearn_version} installed.")
#     sys.exit(1)

# Proceed to load your model
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)


# Load your model
# model = pickle.load(open('model.pkl', 'rb'))

app = Dash(__name__)


app.layout = html.Div([
    html.H1("Credit Card Approval Prediction Form"),
    
    html.Label('Applicant Gender:'),
    dcc.Dropdown(id='input-gender', options=[
        {'label': 'Male', 'value': 0},
        {'label': 'Female', 'value': 1}
    ], value=0),

    html.Label('Owned Car:'),
    dcc.RadioItems(id='input-owned-car', options=[
        {'label': 'Yes', 'value': 1},
        {'label': 'No', 'value': 0}
    ], value=0),

    html.Label('Owned Realty:'),
    dcc.RadioItems(id='input-owned-realty', options=[
        {'label': 'Yes', 'value': 1},
        {'label': 'No', 'value': 0}
    ], value=1),

    html.Label('Total Children:'),
    dcc.Input(id='input-total-children', type='number', value=0),

    html.Div([
        html.Label('Total Income:'),
        dcc.Input(id='input-total-income', type='number', value=180000),
    ]),

    html.Div([
        html.Label('Income Type:'),
        dcc.Dropdown(id='input-income-type', options=[
            {'label': 'Working', 'value': 4},
            {'label': 'State Servant', 'value': 2},
            {'label': 'Commercial Associate', 'value': 0},
            {'label': 'Pensioner', 'value': 1},
            {'label': 'Student', 'value': 3}
        ], value=4),
    ]),

    html.Label('Education Type:'),
    dcc.Dropdown(id='input-education-type', options=[
        {'label': 'Higher education', 'value': 1},
        {'label': 'Secondary / secondary special', 'value': 4},
        {'label': 'Incomplete higher', 'value': 2},
        {'label': 'Lower secondary', 'value': 3},
        {'label': 'Academic degree', 'value': 0}
    ], value=4),

    html.Label('Family Status:'),
    dcc.Dropdown(id='input-family-status', options=[
        {'label': 'Married', 'value': 1},
        {'label': 'Single / not married', 'value': 3},
        {'label': 'Civil marriage', 'value': 0},
        {'label': 'Separated', 'value': 2},
        {'label': 'Widow', 'value': 4}
    ], value=1),

    html.Label('Housing Type:'),
    dcc.Dropdown(id='input-housing-type', options=[
        {'label': 'House / apartment', 'value': 1},
        {'label': 'With parents', 'value': 5},
        {'label': 'Municipal apartment', 'value': 2},
        {'label': 'Rented apartment', 'value': 4},
        {'label': 'Office apartment', 'value': 3},
        {'label': 'Co-op apartment', 'value': 0}
    ], value=1),

    html.Label('Owned Mobile Phone:'),
    dcc.Checklist(id='input-owned-mobile-phone', options=[
        {'label': 'Yes', 'value': 1}
    ], value=[1]),

    html.Label('Owned Work Phone:'),
    dcc.Checklist(id='input-owned-work-phone', options=[
        {'label': 'Yes', 'value': 1}
    ], value=[]),

    html.Label('Owned Phone:'),
    dcc.Checklist(id='input-owned-phone', options=[
        {'label': 'Yes', 'value': 1}
    ], value=[]),

    html.Label('Owned Email:'),
    dcc.Checklist(id='input-owned-email', options=[
        {'label': 'Yes', 'value': 1}
    ], value=[]),

    html.Label('Job Title:'),
    dcc.Dropdown(id='input-job-title', options=[
        {'label': 'Laborers', 'value': 1},
        {'label': 'Core staff', 'value': 2},
        {'label': 'Sales staff', 'value': 3},
        {'label': 'Managers', 'value': 4},
        {'label': 'Drivers', 'value': 5},
        {'label': 'High skill tech staff', 'value': 6},
        {'label': 'Accountants', 'value': 7},
        {'label': 'Medicine staff', 'value': 8},
        {'label': 'Cooking staff', 'value': 9},
        {'label': 'Security staff', 'value': 10},
        {'label': 'Cleaning staff', 'value': 11},
        {'label': 'Private service staff', 'value': 12},
        {'label': 'Low-skill Laborers', 'value': 13},
        {'label': 'Waiters/barmen staff', 'value': 14},
        {'label': 'Secretaries', 'value': 15},
        {'label': 'HR staff', 'value': 16},
        {'label': 'Realty agents', 'value': 17},
        {'label': 'IT staff', 'value': 0}
        
    ], value=1),

    html.Label('Total Family Members:'),
    dcc.Input(id='input-total-family-members', type='number', value=2),

    html.Div([
        html.Label('Applicant Age:'),
        dcc.Input(id='input-applicant-age', type='number', value=30),
    ]),

    html.Div([
        html.Label('Years of Working:'),
        dcc.Input(id='input-years-of-working', type='number', value=5),
    ]),

    html.Div([
        html.Label('Total Bad Debt:'),
        dcc.Input(id='input-total-bad-debt', type='number', value=0),
    ]),

    html.Div([
        html.Label('Total Good Debt:'),
        dcc.Input(id='input-total-good-debt', type='number', value=7),
    ]),

    html.Button('Submit', id='submit-button', n_clicks=0),
    html.Div(id='prediction-output')
])

@app.callback(
    Output('prediction-output', 'children'),
    [Input('submit-button', 'n_clicks')],
    [State('input-gender', 'value'),
     State('input-owned-car', 'value'),
     State('input-owned-realty', 'value'),
     State('input-total-children', 'value'),
     State('input-total-income', 'value'),
     State('input-income-type', 'value'),
     State('input-education-type', 'value'),
     State('input-family-status', 'value'),
     State('input-housing-type', 'value'),
     State('input-owned-mobile-phone', 'value'),
     State('input-owned-work-phone', 'value'),
     State('input-owned-phone', 'value'),
     State('input-owned-email', 'value'),
     State('input-job-title', 'value'),
     State('input-total-family-members', 'value'),
     State('input-applicant-age', 'value'),
     State('input-years-of-working', 'value'),
     State('input-total-bad-debt', 'value'),
     State('input-total-good-debt', 'value')]
)
def predict(n_clicks, gender, owned_car, owned_realty, total_children, total_income, income_type,
            education_type, family_status, housing_type, owned_mobile_phone, owned_work_phone, owned_phone,
            owned_email, job_title, total_family_members, applicant_age, years_of_working,
            total_bad_debt, total_good_debt):
    if n_clicks > 0:
        # Assume model is loaded and named 'model'
        # Here you would convert these inputs into the format your model expects
        # For simplicity, assuming all inputs are used directly
        input_df = pd.DataFrame([[
            gender, owned_car, owned_realty, total_children, total_income, income_type,
            education_type, family_status, housing_type, owned_mobile_phone[0] if owned_mobile_phone else 0,
            owned_work_phone[0] if owned_work_phone else 0, owned_phone[0] if owned_phone else 0,
            owned_email[0] if owned_email else 0, job_title, total_family_members, applicant_age,
            years_of_working, total_bad_debt, total_good_debt
        ]], columns=['Applicant_Gender', 'Owned_Car', 'Owned_Realty', 'Total_Children', 'Total_Income', 'Income_Type',
                     'Education_Type', 'Family_Status', 'Housing_Type', 'Owned_Mobile_Phone', 'Owned_Work_Phone',
                     'Owned_Phone', 'Owned_Email', 'Job_Title', 'Total_Family_Members', 'Applicant_Age',
                     'Years_of_Working', 'Total_Bad_Debt', 'Total_Good_Debt'])
        prediction = model.predict(input_df)
        return f'Prediction: {"Approved" if prediction[0] == 1 else "Not Approved"}'
    return 'Enter your data and press submit.'

if __name__ == '__main__':
    app.run_server(debug=True)





# app.layout = html.Div([
#     html.Div(children='Hello World')
# ])



# if __name__ == '__main__':
#     app.run(debug=True)