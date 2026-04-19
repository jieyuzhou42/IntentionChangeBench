

## WebShop Dataset Size

The simulation runner defaults to the lightweight 1k-product WebShop subset:

```powershell
python .\src\run_simulation.py --webshop_num_products 1000
```

To use the full WebShop product dataset, first make sure these files exist:

- `WebShop/data/items_shuffle.json`
- `WebShop/data/items_ins_v2_1000.json`
- `WebShop/search_engine/indexes/`

Then run:

```powershell
python .\src\run_simulation.py --webshop_num_products all
```

`--webshop_num_products 100000` is also supported if you built `WebShop/search_engine/indexes_100k/`.

The product catalog and instruction/attribute file can be scaled separately.
By default, `--webshop_num_products all` uses the full product catalog with the
existing 1k instruction/attribute file. Set `WEBSHOP_ATTR_DATASET=all` only if
you also downloaded `WebShop/data/items_ins_v2.json`.
