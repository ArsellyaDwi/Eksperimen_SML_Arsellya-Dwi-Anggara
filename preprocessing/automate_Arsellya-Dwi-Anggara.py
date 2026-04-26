import pandas as pd

def preprocess_data(input_path, output_path):
    # Load data
    df = pd.read_csv(input_path)

    # =======================
    # Missing Values
    # =======================
    df['Age'] = df['Age'].fillna(df['Age'].median())
    df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

    # Drop Cabin
    if 'Cabin' in df.columns:
        df = df.drop(columns=['Cabin'])

    # =======================
    # Drop kolom tidak penting
    # =======================
    drop_cols = ['PassengerId', 'Name', 'Ticket']
    df = df.drop(columns=[col for col in drop_cols if col in df.columns])

    # =======================
    # Encoding
    # =======================
    df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})

    df = pd.get_dummies(df, columns=['Embarked'], drop_first=True)

    # =======================
    # Feature Engineering
    # =======================
    df['FamilySize'] = df['SibSp'] + df['Parch']

    # =======================
    # Save hasil
    # =======================
    df.to_csv(output_path, index=False)

    print("Preprocessing selesai! File disimpan di:", output_path)


# Run manual (biar bisa langsung jalan)
if __name__ == "__main__":
    preprocess_data(
        input_path="dataset_raw/train.csv",
        output_path="preprocessing/titanic_preprocessing.csv"
    )