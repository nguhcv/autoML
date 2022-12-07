import json

data = {
    "dataset":
        {
            "type": "univariate",
            "path": "C:/Project/autoML/auto_ML/dataset/power_demand/full_series.csv",
            "column_name": [ "time", "value", "label"],
            "input_column": "value"
        },

    "model": "sarima",
    "h_param": {
        "order": [(1, 1, 1), (1, 1, 2), (1, 2, 1), (1, 2, 2), (1, 2, 3), (1, 3, 2), (1, 3, 3),
                  (2, 1, 1), (2, 1, 2)],
        "seasonal_order": [(1, 1, 1, 7), (1, 1, 1, 14), (1, 1, 1, 30), (1, 1, 1, 60), (1,1,1,51)]


    },
    "metric": "mae"
}

with open('ad_test.json', 'w', encoding='utf-8') as file:
  json.dump(data, file, indent="\t")
