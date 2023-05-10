import os

gene_folder = "/home/hdo/Data/genetics/EGAD00010001226/001"
temp_folder = "/home/hdo/BIM-BED-FAM"
your_id = "hdo"

for file in os.listdir(gene_folder):
    if file.endswith("bed") or file.endswith("bim") or file.endswith("fam"):
        new_name = ""
        if file.endswith("bed"):
            chr_number = file.split("_")[1][1:]
            new_name = "ukb_{}_chr{}.bed".format(your_id, chr_number)
        elif file.endswith("bim"):
            chr_number = file.split("_")[2][3:]
            new_name = "ukb_{}_chr{}.bim".format(your_id, chr_number)
        else:
            chr_number = file.split("_")[1][1:]
            new_name = "ukb_{}_chr{}.fam".format(your_id, chr_number)

        file_full_path = os.path.join(gene_folder, file)
        linked_full_path = os.path.join(temp_folder, new_name)
        if (new_name in os.listdir(temp_folder)):
            continue
        # command = "ln -s {} {}".format(
        #     file_full_path, linked_full_path)

        os.symlink(file_full_path, linked_full_path)
