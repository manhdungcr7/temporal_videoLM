import shutil
shutil.copyfile(OUT/'uploads/cq_diagnostic_manifest_s1.json',OUT/'cq_diagnostic_manifest_s1.json')
shutil.copyfile(OUT/'uploads/cq_diagnostic_round6_s1.jsonl',OUT/'cq_diagnostic_round6_s1.jsonl')
for index in (1454,2396,1727,1647,2345):
    shutil.copyfile(OUT/f'uploads/cq_physical_memory_{index}.json',OUT/f'cq_physical_memory_{index}.json')
