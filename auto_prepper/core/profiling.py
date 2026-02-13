import polars as pl

from auto_prepper.core.dataset import Dataset
from auto_prepper.utils.feature_type import FeatureType
from auto_prepper.utils.str_subtype import StrSubtype


class Profile:
    _outliers_threshold = 1.5
    _imbalance_threshold = 1.5

    def __init__(self, ds, duplicate_approx=False, n_rows_to_check=0):
        self._df = ds.df
        self._target_column = ds.target_column
        self._duplicate_approx = duplicate_approx
        self._n_rows_to_check = n_rows_to_check
        self._profile = self._run()
        self._profile.update(self._run_outliers_imbalance())
        self._profile['summary'] = self._summarize()

    @property
    def profile(self):
        return self._profile

    def _shape(self):
        df = self._df
        n_rows, n_columns = df.shape
        return {'n_rows': n_rows, 'n_columns': n_columns}

    def _data_types(self):
        df = self._df
        data_types = dict(df.schema)
        for col in data_types:
            data_types[col] = data_types[col].base_type()
        data_types['type_count'] = len(set(df.dtypes))
        return data_types

    def _str_subtypes(self):
        df = self._df
        str_subtypes = {col.name: None for col in df}
        str_temporal_columns = StrSubtype.select_temporal(
            df, self._n_rows_to_check
        )
        for col in str_temporal_columns:
            str_subtypes[col.name] = StrSubtype.TEMPORAL
        str_decodable_columns = StrSubtype.select_decodable(
            df, self._n_rows_to_check
        )
        for col in str_decodable_columns:
            str_subtypes[col.name] = StrSubtype.DECODABLE
        str_subtypes['type_count'] = (
            str_temporal_columns.width + str_decodable_columns.width
        )
        return str_subtypes

    def _feature_types(self):
        df = self._df
        types = {col.name: FeatureType.get_feature_type(col) for col in df}
        types['type_count'] = len(set(types.values()))
        return types

    def _missing_values(self):
        df = self._df
        n_rows = df.height
        missing_values = df.null_count().to_dict(as_series=False)
        for col in missing_values:
            missing_values[col] = {
                'count': missing_values[col][0],
                'ratio': missing_values[col][0] / n_rows,
            }
        missing_values['df'] = {
            'count': sum(
                missing_values[col]['count'] for col in missing_values
            ),
        }
        missing_values['df']['ratio'] = missing_values['df']['count'] / (
            n_rows * df.width
        )
        return missing_values

    def _duplicates(self):
        df = self._df
        if self._duplicate_approx:
            combined_column = df.select(
                pl.concat_str(df.columns, separator='|').alias('combined')
            )
            approx_unique_count = combined_column.select(
                pl.col('combined').approx_n_unique()
            )[0, 0]
            count = df.height - approx_unique_count
        else:
            count = df.height - df.n_unique()
        return {'count': count, 'ratio': count / df.height}

    def _numeric_stats(self):
        df = self._df
        df = FeatureType.select(df, FeatureType.NUMERIC)
        stats = {
            col.name: {
                'unique_count': col.unique_counts().shape[0],
                'mean': col.mean(),
                'std': col.std(),
                'min': col.min(),
                'q25': col.quantile(0.25),
                'median': col.median(),
                'q75': col.quantile(0.75),
                'max': col.max(),
            }
            for col in df
        }
        return stats

    def _categorical_stats(self):
        df = self._df
        df = FeatureType.select(df, FeatureType.CATEGORICAL)
        stats = {}
        for col in df:
            unique_counts = col.unique_counts()
            stats[col.name] = {
                'unique_count': unique_counts.shape[0],
                'mean_#': unique_counts.mean(),
                'std_#': unique_counts.std(),
                'min_#': unique_counts.min(),
                'median_#': unique_counts.median(),
                'max_#': unique_counts.max(),
                'mode': unique_counts.mode()[0],
            }
        return stats

    def _outliers(self):
        df = self._df
        count = 0
        for col in self._profile['numeric_stats']:
            min_val = self._profile['numeric_stats'][col]['min']
            max_val = self._profile['numeric_stats'][col]['max']
            q25 = self._profile['numeric_stats'][col]['q25']
            q75 = self._profile['numeric_stats'][col]['q75']
            if q25 is None or q75 is None:
                continue
            iqr = q75 - q25
            if (
                q25 - self._outliers_threshold * iqr > min_val
                or q75 + self._outliers_threshold * iqr < max_val
            ):
                count += 1
        return {'count': count, 'ratio': count / df.width}

    def _imbalance(self):
        if (
            not self._target_column
            or not self._profile['feature_types'][
                self._target_column
            ].is_categorical()
            or self._profile['categorical_stats'][self._target_column]['max_#']
            / self._profile['categorical_stats'][self._target_column]['min_#']
            <= self._imbalance_threshold
        ):
            return False
        return True

    def _run(self):
        profile = {
            'shape': self._shape(),
            'data_types': self._data_types(),
            'str_subtypes': self._str_subtypes(),
            'feature_types': self._feature_types(),
            'missing_values': self._missing_values(),
            'duplicates': self._duplicates(),
            'numeric_stats': self._numeric_stats(),
            'categorical_stats': self._categorical_stats(),
        }
        return profile

    def _run_outliers_imbalance(self):
        profile = {
            'cols_w_outliers': self._outliers(),
            'imbalanced_target': self._imbalance(),
        }
        return profile

    def _summarize(self):
        summary = {
            'shape_n_rows': self._profile['shape']['n_rows'],
            'shape_n_columns': self._profile['shape']['n_columns'],
            'data_types_count': self._profile['data_types']['type_count'],
            'str_subtypes_count': self._profile['str_subtypes']['type_count'],
            'feature_types_count': self._profile['feature_types'][
                'type_count'
            ],
            'missing_values_count': self._profile['missing_values']['df'][
                'count'
            ],
            'duplicates_count': self._profile['duplicates']['count'],
            'cols_w_outliers': self._profile['cols_w_outliers']['count'],
            'imbalanced_target': int(self._profile['imbalanced_target']),
        }
        return summary

    def summary_table(self):
        summary = self._profile['summary']
        df_summary = {
            'summary': summary.keys(),
            'ds': summary.values(),
        }
        df_summary = pl.DataFrame(df_summary)
        return df_summary

    def __str__(self):
        shape = self._profile['shape']
        missing_values = self._profile['missing_values']
        duplicates = self._profile['duplicates']
        data_types = self._profile['data_types']
        str_subtypes = self._profile['str_subtypes']
        feature_types = self._profile['feature_types']
        numeric_stats = self._profile['numeric_stats']
        categorical_stats = self._profile['categorical_stats']
        cols_w_outliers = self._profile['cols_w_outliers']
        imbalance = self._profile['imbalanced_target']

        df_shape = {'shape': shape.keys(), 'ds': shape.values()}
        df_shape = pl.DataFrame(df_shape)

        df_types = {'types': ['data_type', 'str_subtype', 'feature_type']}
        for col in data_types:
            df_types[col] = [
                data_types[col],
                str_subtypes[col],
                feature_types[col],
            ]
        df_types = pl.DataFrame(df_types)

        df_missing_values = {
            'missing_values': list(missing_values.values())[0].keys()
        }
        for col in missing_values:
            df_missing_values[col] = missing_values[col].values()
        df_missing_values = pl.DataFrame(df_missing_values)

        df_duplicates = {
            'duplicates': duplicates.keys(),
            'ds': duplicates.values(),
        }
        df_duplicates = pl.DataFrame(df_duplicates)

        df_numeric_stats = {'numeric_stats': []}
        if numeric_stats:
            df_numeric_stats['numeric_stats'] = list(numeric_stats.values())[
                0
            ].keys()
        for col in numeric_stats:
            df_numeric_stats[col] = numeric_stats[col].values()
        df_numeric_stats = pl.DataFrame(df_numeric_stats)

        df_categorical_stats = {'categorical_stats': []}
        if categorical_stats:
            df_categorical_stats['categorical_stats'] = list(
                categorical_stats.values()
            )[0].keys()
        for col in categorical_stats:
            df_categorical_stats[col] = categorical_stats[col].values()
        df_categorical_stats = pl.DataFrame(df_categorical_stats)

        df_outliers = {
            'cols_w_outliers': cols_w_outliers.keys(),
            'ds': cols_w_outliers.values(),
        }
        df_outliers = pl.DataFrame(df_outliers)

        df_imbalance = {
            'imbalanced_target': imbalance,
        }
        df_imbalance = pl.DataFrame(df_imbalance)

        df_summary = self.summary_table()

        pl.Config.set_tbl_hide_dataframe_shape(True)

        s = '\n'.join(
            [
                str(df_shape),
                str(df_types),
                str(df_missing_values),
                str(df_duplicates),
                str(df_numeric_stats),
                str(df_categorical_stats),
                str(df_outliers),
                str(df_imbalance),
                str(df_summary),
            ]
        )

        pl.Config.set_tbl_hide_dataframe_shape(False)

        return s

    def __repr__(self):
        return self.__str__()
