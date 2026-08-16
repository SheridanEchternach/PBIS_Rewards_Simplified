import joblib
import pandas as pd
from app.model import predict_reinforcement

model=joblib.load("model/student_model.pkl")

test_student = pd.DataFrame({
    "points_this_week":[3, 5, 7],
    "attendance_rate":[91, 89, 99],
    "behavior_referrals":[2, 0, 1]
})

prediction=model.predict(test_student)


students = [
    (3,91,2),
    (5, 89, 0),
    (7,99,1)
]

for points, attendance, referrals in students:
    result = predict_reinforcement(points, attendance, referrals)

    print("Student:")
    print(f"Points: {points}")
    print(f"Attendance: {attendance}")
    print(f"Behavior referrals: {referrals}")
    print(result)

print(test_student)
print(prediction)