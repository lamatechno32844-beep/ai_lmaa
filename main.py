# main.py
# Predict student major (stu_group)

import joblib
import pandas as pd


def main():

    # تحميل الموديل
    model = joblib.load("random_forest_model.pkl")

    # تحميل أعمدة التدريب
    model_columns = joblib.load("model_columns.pkl")

    # قراءة البيانات
    data = pd.read_csv("student_performance_new_columns.csv")

    # حذف العمود الحقيقي إذا موجود
    if "stu_group" in data.columns:
        X = data.drop("stu_group", axis=1)
    else:
        X = data.copy()

    # Encoding
    X = pd.get_dummies(X)

    # إضافة الأعمدة الناقصة
    for col in model_columns:
        if col not in X.columns:
            X[col] = 0

    # ترتيب الأعمدة مثل التدريب
    X = X[model_columns]

    # التوقع
    predictions = model.predict(X)

    # إضافة التخصص المتوقع
    data["Predicted_stu_group"] = predictions

    # عرض أول 5 نتائج
    print("\nStudent Major Classification Results:\n")
    print(data[["Predicted_stu_group"]].head())

    # حفظ النتائج
    data.to_csv("classified_students.csv", index=False)

    print("\nResults saved to classified_students.csv")


if __name__ == "__main__":
    main()