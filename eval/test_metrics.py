import os
import sys
from datetime import datetime

import cv2
from tqdm import tqdm
from metrics import MAE, Emeasure, Fmeasure, Smeasure, WeightedFmeasure

preds_data_root = sys.argv[1] if len(sys.argv) > 1 else "./res/D2GF-Net/final/"
f = open(preds_data_root+"/log.txt","a+",encoding="UTF-8")

results_table_path = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "test_results.txt",
)
table_rows = {}

for _data_name in ['CHAMELEON','CAMO', 'COD10K','NC4K' ]:

    FM = Fmeasure()
    WFM = WeightedFmeasure()
    SM = Smeasure()
    EM = Emeasure()
    mae = MAE()
    masks_data_root = "./data/TestDataset/" + _data_name

    mask_root = os.path.join(masks_data_root, "GT")
    pred_root = os.path.join(preds_data_root, _data_name)
    mask_name_list = sorted(os.listdir(mask_root))
    for mask_name in tqdm(mask_name_list, total=len(mask_name_list)):
        mask_path = os.path.join(mask_root, mask_name)
        pred_path = os.path.join(pred_root, mask_name)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        pred = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)
        if mask.shape != pred.shape:
            pred = cv2.resize(pred, (mask.shape[1], mask.shape[0]), interpolation=cv2.INTER_LINEAR)
        FM.step(pred=pred, gt=mask)
        WFM.step(pred=pred, gt=mask)
        SM.step(pred=pred, gt=mask)
        EM.step(pred=pred, gt=mask)
        mae.step(pred=pred, gt=mask)

    fm = FM.get_results()["fm"]
    wfm = WFM.get_results()["wfm"]
    sm = SM.get_results()["sm"]
    em = EM.get_results()["em"]
    mae = mae.get_results()["mae"]

    results = {
        "Smeasure": sm,
        "wFmeasure": wfm,
        "MAE": mae,
        "adpEm": em["adp"],
        "meanEm": em["curve"].mean(),
        "maxEm": em["curve"].max(),
        "adpFm": fm["adp"],
        "meanFm": fm["curve"].mean(),
        "maxFm": fm["curve"].max(),
    }

    print(results)
    f.write(_data_name+"\n")
    f.write(str(results) + "\n\n")
    f.flush()

    table_rows[_data_name] = (sm, wfm, em["curve"].mean(), mae)


f.close()

with open(results_table_path, "a", encoding="UTF-8") as tf:
    tf.write("\n==== Test {} ({}) ====\n".format(
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"), preds_data_root))
    tf.write("Dataset     S_a    wFm    mEm    MAE\n")
    for name in ['CAMO', 'CHAMELEON', 'COD10K', 'NC4K']:
        if name not in table_rows:
            continue
        sm, wfm, mem, mae = table_rows[name]
        tf.write("{:<11s}{:.3f}  {:.3f}  {:.3f}  {:.3f}\n".format(
            name, sm, wfm, mem, mae))

print("\nResults table appended to: " + results_table_path)
