import pandas as pd

# 1️⃣ Cargar datasets originales
trans = pd.read_csv("test_transaction.csv")
ident = pd.read_csv("test_identity.csv")

# 2️⃣ Tomar muestra de 100 transacciones aleatorias
sample_trans = trans.sample(n=100, random_state=42)

# 3️⃣ Merge con identidad
merged = sample_trans.merge(ident, on="TransactionID", how="left")

# 4️⃣ Eliminar columna objetivo si existe
if "isFraud" in merged.columns:
    merged = merged.drop(columns=["isFraud"])

# 5️⃣ Guardar como CSV de nuevas transacciones
file_path = "new_transactions.csv"
merged.to_csv(file_path, index=False)

file_path
