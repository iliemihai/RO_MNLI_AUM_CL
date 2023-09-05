import pandas as pd
import numpy as np
import ast

data = pd.read_csv("../train_romnli.tsv", sep="\t")

df = pd.read_csv("full_aum_records.csv")
epoch = 5

df_class_0 = df[(df["target_logit"] == 0) & (df["num_measurements"] == epoch)]
mean_class_0 = df_class_0["target_val"].values.mean()

df_class_1 = df[(df["target_logit"] == 1) & (df["num_measurements"] == epoch)]
mean_class_1 = df_class_1["target_val"].values.mean()

df_class_2 = df[(df["target_logit"] == 2) & (df["num_measurements"] == epoch)]
mean_class_2 = df_class_2["target_val"].values.mean()

C_matrix = np.zeros((3, 3))
Q_matrix = np.zeros((3, 3))

dff = df[df["num_measurements"] == epoch]

for cls, logits in zip(dff["target_logit"].values, dff["logits"].values):
    logits = ast.literal_eval(logits)
    if logits[0] > mean_class_0:
        C_matrix[cls, 0] += 1
    elif logits[1] > mean_class_1:
        C_matrix[cls, 1] += 1
    elif logits[2] > mean_class_2:
        C_matrix[cls, 2] += 1

def normalizeMatrix(M):
    norm = 0
    for row in M:
        for val in row:
            norm += val
    return M / norm

Q_matrix = normalizeMatrix(C_matrix)

print("C matrix: ", C_matrix)
print("Q matrix: ", Q_matrix)


g = open("mislabeled_neutral.txt", "w")
df_class_0_sorted = df_class_0.sort_values(by='aum')
percentage_mislabeled_0 = Q_matrix[0,1] + Q_matrix[0,2]
num_rows_0 = int(len(df_class_0_sorted) * percentage_mislabeled_0)
print("NUM MISLABELED NEUTRAL: ",num_rows_0)
selected_rows_0 = df_class_0_sorted.head(num_rows_0)
selected_ids_0 = selected_rows_0["sample_id"].values
for id, sent1, sent2 in zip(data["guid"], data["sentence1"], data["sentence2"]):
    if id in selected_ids_0:
        g.write(sent1+"\t\t"+sent2+"\n")
    

g = open("mislabeled_entailment.txt", "w")
df_class_1_sorted = df_class_1.sort_values(by='aum')
percentage_mislabeled_1 = Q_matrix[1,0] + Q_matrix[1,2]
num_rows_1 = int(len(df_class_1_sorted) * percentage_mislabeled_1)
print("NUM MISLABELED ENTAILMENT: ",num_rows_1)
selected_rows_1 = df_class_1_sorted.head(num_rows_1)
selected_ids_1 = selected_rows_1["sample_id"].values
for id, sent1, sent2 in zip(data["guid"], data["sentence1"], data["sentence2"]):
    if id in selected_ids_1:
        g.write(sent1+"\t\t"+sent2+"\n")
 

g = open("mislabeled_contradiction.txt", "w")
df_class_2_sorted = df_class_2.sort_values(by='aum')
percentage_mislabeled_2 = Q_matrix[2,0] + Q_matrix[2,1]
num_rows_2 = int(len(df_class_2_sorted) * percentage_mislabeled_2)
print("NUM MISLABELED CONTRADICTION: ",num_rows_2)
selected_rows_2 = df_class_2_sorted.head(num_rows_2)
selected_ids_2 = selected_rows_2["sample_id"].values
for id, sent1, sent2 in zip(data["guid"], data["sentence1"], data["sentence2"]):
    if id in selected_ids_2:
        g.write(sent1+"\t\t"+sent2+"\n")
 
