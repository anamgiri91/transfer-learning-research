# Data licences

Record every source before use. The code in this repo is MIT; the data is not.

| Source | Licence | Redistribution | Attribution required |
|---|---|---|---|
| ChEMBL | CC BY-SA 3.0 | Yes, share-alike | Yes — cite the release version |
| PubChem | Public domain | Yes | Courtesy citation |
| BindingDB | CC BY 3.0 | Yes | Yes |
| PDB | CC0 | Yes | Courtesy citation |
| Literature-curated values | Per publisher | **Check individually** | Yes |
| Pretrained encoder weights | Per model card | Check before redistributing | Yes |

Derived files in `data/processed/` inherit the most restrictive licence among
their inputs. ChEMBL's share-alike term propagates: a released benchmark built
on it must carry CC BY-SA.
