import joblib
import pandas as pd

model=joblib.load("model/student_model.pkl")

test_student = pd.DataFrame({
    "points_this_week":[3, 5, 7],
    "attendance_rate":[91, 89, 99],
    "behavior_referrals":[2, 0, 1]
})

prediction=model.predict(test_student)


print(test_student)
print(prediction)