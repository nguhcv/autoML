import json

data = {
    "dataset":
        {
            "type": "univariate",
            "path": "C:/Project/autoML/auto_ML/dataset/air_passengers.csv",
            "column_name": ["time", "value"],
            "input_column": "value"
        },

    "preprocessing": {
        "handling_missing": ["mean", "median"],
    },

    "model": "sarima",
    "h_param": {
        "order": [(1, 1, 1), (1, 1, 2), (1, 1, 3), (1, 2, 1), (1, 2, 2), (1, 2, 3), (1, 3, 1), (1, 3, 2), (1, 3, 3),
                  (2, 1, 1), (2, 1, 2)],
        "seasonal_order": [(1, 1, 1, 12), (1, 1, 1, 24), (1, 1, 1, 30), (1, 1, 1, 60)]

    },
    "metric": "mae"
}

with open('fc_test.json', 'w', encoding='utf-8') as file:
  json.dump(data, file, indent="\t")
