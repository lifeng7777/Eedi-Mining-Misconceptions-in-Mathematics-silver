import json

with open("/data/train_data_with_ref.json", 'r') as f:
    data1 = json.load(f)


with open("/data/train_data_with_new_question_ref.json", 'r') as f:
    data2 = json.load(f)

final_data = data1 + data2

with open("/data/train_data_with_ref_all.json", 'w') as f:
    f.write(json.dumps(final_data, indent=2, ensure_ascii=False))
