import polars as pl

from auto_prepper.io.input import parse_raw_data
from auto_prepper.io.integration import df_join_diagonal
from auto_prepper.io.output import save_df
from auto_prepper.io.separation import (
    train_test_split,
    train_test_split_stratified,
)
from auto_prepper.utils.dataset_type import DatasetType
from auto_prepper.utils.exceptions import DatasetTypeError, TargetColumnError


class Dataset:
    _str_hide_dataframe_meta_info = False

    def __init__(
        self, *data_sources, dataset_type='train', target_column=None
    ):
        self._df = self._parse_integrate_raw_data(*data_sources)
        self._set_dataset_type(dataset_type)
        self._set_target_column(target_column)

    @classmethod
    def set_str_hide_df_non_tbl(cls, hide=False):
        # hide dataframe non-table info in str format
        cls._str_hide_dataframe_meta_info = hide
        pl.Config.set_tbl_hide_dataframe_shape(hide)

    @property
    def df(self):
        return self._df

    @property
    def dataset_type(self):
        return self._dataset_type

    @property
    def target_column(self):
        return self._target_column

    @df.setter
    def df(self, df):
        if not isinstance(df, pl.DataFrame):
            raise TypeError(
                'Dataset df must be of type polars.DataFrame, cannot set to '
                + f'value of type {type(df)}'
            )
        if self._target_column and self._target_column not in df.columns:
            raise TargetColumnError(self._target_column)
        self._df = df

    def _set_dataset_type(self, dataset_type):
        if isinstance(dataset_type, str):
            dataset_type = DatasetType.from_str(dataset_type)
        if dataset_type not in [
            DatasetType.TRAIN,
            DatasetType.TEST,
            DatasetType.INFERENCE,
        ]:
            raise DatasetTypeError(expected_type='train, test or inference')
        self._dataset_type = dataset_type

    def _set_target_column(self, target_column):
        if target_column and target_column not in self._df.columns:
            raise TargetColumnError(target_column)
        self._target_column = target_column

    def _parse_integrate_raw_data(self, *data_sources):
        dfs = [parse_raw_data(data_source) for data_source in data_sources]
        df = df_join_diagonal(*dfs)
        return df

    def train_test_split(self, shuffle=True):
        split_done = False
        if self._target_column:
            try:
                df_train, df_test = train_test_split_stratified(
                    df=self.df, columns=self._target_column, shuffle=shuffle
                )
                split_done = True
            except Exception:
                print(
                    'Stratified train test split failed, performing random '
                    + 'split instead.'
                )
        if not split_done:
            df_train, df_test = train_test_split(df=self.df, shuffle=shuffle)
        ds_train = Dataset(
            df_train,
            dataset_type='train',
            target_column=self._target_column,
        )
        ds_test = Dataset(
            df_test,
            dataset_type='test',
            target_column=self._target_column,
        )
        return ds_train, ds_test

    def copy(self):
        ds = Dataset(
            self._df.clone(),
            dataset_type=str(self._dataset_type),
            target_column=self._target_column,
        )
        return ds

    def save_df(self, format, path):
        save_df(self.df, format, path)

    def __getattr__(self, name):
        # forwarding attribute and method access to self._df
        # for those undefined in class
        dataframe = object.__getattribute__(self, '_df')
        return getattr(dataframe, name)

    def __str__(self):
        s = str(self._df)
        if not Dataset._str_hide_dataframe_meta_info:
            s = (
                f'dataset_type: {self.dataset_type}\n'
                + f'target_column: {self.target_column}\n'
                + s
            )
        return s

    def __repr__(self):
        return self.__str__()
