import re



def match_license(lp_str):
    lp_str = re.sub(r'\W+', '', lp_str)
    regex = r"\d{2}[A-Z]{1}\d{4,5}"
    is_match = re.findall(regex, lp_str, re.MULTILINE)
    if not is_match:
        return None
    else:
        return " ".join(is_match)

test_str = "223432451EF6322sdfasdfasdf"

# is_match = match_license(test_str)
# print(is_match)



import csv
import re
import sys

with open("gen_data/annot_file.csv") as source,  open("gen_data/ano.csv", "w", newline="") as result:
    reader = csv.reader(source)
    writer = csv.writer(result)
    regex = r"(\d{2}[A-Z])(\d{3})(\d{2})"
    for row in reader:
        i = row[3]
        is_match = re.findall(regex, i, re.MULTILINE)
        if is_match:
            print(is_match)
            row[3] = '{}-{}.{}'.format(is_match[0][0], is_match[0][1], is_match[0][2])
            writer.writerow(row)
        else:
            if len(i) > 7:
                writer.writerow(row)
