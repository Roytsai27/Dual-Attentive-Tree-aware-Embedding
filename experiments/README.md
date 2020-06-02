## Code for Auxiliary Experiments

In this directory, we provide codes and bash scripts for running auxiliary experiments:
* [revcls](./kdd2020-exp-revcls/): Section 5.1, date_cls and date_rev results 
* [ablation-studies](./kdd2020-exp-ablation-studies/): Section 5.3, includes w/o attention network and w/o fusion. Modify `model/AttTreeEmbedding.py` with the provided code. w/o dual task learning and w/o multi-head self attention could be done by setting args in `train.py`
* [training-length](./kdd2020-exp-training-length/): Section 5.4, effects on training length
* [corrupted-data](./kdd2020-exp-corrupted-data/): Section 6, way to leverage existing data
* [hyperparameter-analysis](./kdd2020-exp-hyperparameter-analysis): Section 7.1-2, hyperparameter analysis
* [loss-weight](./kdd2020-exp-loss-weight): Section 7.3, date_cls and date_rev by controlling alpha
* [interpreting-results](./Interpreting-DATE-Results.ipynb): Section 5.6, interpreting DATE results by finding effective cross-features with high attention weight 

---

Note that there are minor differences of variable names used in these codes.

**Features**
* 'SGD.NAME' → 'sgd.id'
* 'SGD.DATE' → 'sgd.date'
* 'IMPORTER.TIN' → 'importer.id'
* 'DECLARANT.CODE' → 'declarant.id'
* 'ORIGIN.CODE' → 'country'
* 'OFFICE' → 'office.id'
* 'TARIFF.CODE' → 'tariff.code'
* 'QUANTITY' → 'quantity'
* 'GROSS.WEIGHT' → 'gross.weight'
* 'FOB.VALUE' → 'fob.value'
* 'CIF.VALUE' → 'cif.value'
* 'TOTAL.TAXES' → 'total.taxes'

**Labels**
* 'illicit' → 'illicit'
* 'RAISED_TAX_AMOUNT' → 'revenue'

**Removed Features**
* 'RECEIPT.DATE'
