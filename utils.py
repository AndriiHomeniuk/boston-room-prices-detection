from pyspark.ml.feature import OneHotEncoder, StringIndexer


def update_columns(column_list, main_df):
    for column_name in column_list:
        string_indexer = StringIndexer(inputCol=column_name, outputCol=f'{column_name}_Index')
        model = string_indexer.fit(main_df)
        indexed = model.transform(main_df)

        encoder = OneHotEncoder(inputCol=f'{column_name}_Index', outputCol=f'{column_name}_Vec')
        main_df = encoder.transform(indexed)
    return main_df
