import joblib
import pandas as pd

model=joblib.load("model/student_model.pkl")

def predict_reinforcement (points_this_week, attendance_rate, behavior_referrals):
    data= pd.DataFrame(
        [[
            points_this_week,
            attendance_rate,
            behavior_referrals
        ]],
        columns=[
            "points_this_week",
            "attendance_rate",
            "behavior_referrals"
        ]
    )

    prediction = model.predict(data)[0]
    probabilities = model.predict_proba(data)[0]
    probability = probabilities[prediction]

    if prediction == 1:
        result= "Reinforcement Needed"
    else:
        result= "No Immediate Concern"

    return{
        "needs_reinforcement": bool(prediction ==1),
        "prediction": result,
        "confidence": round(float(probability),2)
    }