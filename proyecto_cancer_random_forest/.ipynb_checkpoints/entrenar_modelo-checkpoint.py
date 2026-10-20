"""Entrena un modelo Random Forest para el conjunto Wisconsin de cáncer de mama.
Ejecutar una vez antes de iniciar la API: python entrenar_modelo.py
Este proyecto es académico y no sirve para diagnosticar pacientes reales.
"""
from pathlib import Path
import joblib
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

CARPETA = Path(__file__).resolve().parent

datos = load_breast_cancer(as_frame=True)
X = datos.data
y = datos.target  # 0 = maligno, 1 = benigno en este conjunto de scikit-learn

X_entrenamiento, X_prueba, y_entrenamiento, y_prueba = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

modelo = RandomForestClassifier(
    n_estimators=300,
    random_state=42,
    class_weight="balanced",
    max_features="sqrt"
)
modelo.fit(X_entrenamiento, y_entrenamiento)
predicciones = modelo.predict(X_prueba)

joblib.dump(modelo, CARPETA / "modelo_cancer.pkl")
joblib.dump(list(X.columns), CARPETA / "columnas_modelo.pkl")

print(f"Exactitud en prueba: {accuracy_score(y_prueba, predicciones) * 100:.2f}%")
print("Matriz de confusión:")
print(confusion_matrix(y_prueba, predicciones))
print("Reporte de clasificación:")
print(classification_report(y_prueba, predicciones, target_names=["Maligno", "Benigno"]))
print("Archivos guardados en:", CARPETA)
