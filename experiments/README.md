## Code for Auxiliary Experiments

In this directory, we provide codes and bash scripts for running auxiliary experiments:
* revcls: Section 5.1, date_cls and date_rev results 
* training-length: Section 5.4, effects on training length
* corrupted-data: Section 6, way to leverage existing data
* hyperparameter-analysis: Section 7.1-2, hyperparameter analysis
* loss-weight: Section 7.3, date_cls and date_rev by controlling alpha
* [cross-feature](Extracting-cross-features-from-attention-network.ipynb): Section 5.6, finding effective cross features with high attention weight 

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
