from enum import Enum

from auto_prepper.utils.data_types import (
    INTEGER_TYPES,
    NUMERIC_TYPES,
    STRING_TYPES,
)


class FeatureType(Enum):
    _ordinal_threshold = 0.1
    ANY = 0
    NUMERIC = 1
    NUMERIC_DISCRETE = 2
    NUMERIC_CONTINUOUS = 3
    CATEGORICAL = 4
    CATEGORICAL_ORDINAL = 5
    CATEGORICAL_NOMINAL = 6
    NUMERIC_OR_ORDINAL = 7
    OTHER = 8

    @classmethod
    def get_feature_type(cls, df_col):
        col_dtype = df_col.dtype
        if col_dtype in NUMERIC_TYPES:
            if col_dtype in INTEGER_TYPES:
                return FeatureType.NUMERIC_DISCRETE
            else:
                return FeatureType.NUMERIC_CONTINUOUS
        if col_dtype in STRING_TYPES:
            return FeatureType.CATEGORICAL_NOMINAL
        return FeatureType.OTHER

    @staticmethod
    def select(df, feature_type):
        columns = []
        for col in df:
            if feature_type.match_or_supertype_col(col):
                columns.append(col.name)
        return df.select(columns)

    def is_numeric(self):
        return self in {
            FeatureType.NUMERIC,
            FeatureType.NUMERIC_DISCRETE,
            FeatureType.NUMERIC_CONTINUOUS,
        }

    def is_categorical(self):
        return self in {
            FeatureType.CATEGORICAL,
            FeatureType.CATEGORICAL_ORDINAL,
            FeatureType.CATEGORICAL_NOMINAL,
        }

    def is_numeric_or_ordinal(self):
        return self.is_numeric() or self in {
            FeatureType.CATEGORICAL_ORDINAL,
            FeatureType.NUMERIC_OR_ORDINAL,
        }

    def match_or_supertype(self, feature_type):
        if self == FeatureType.ANY:
            return True
        if self == feature_type:
            return True
        if self == FeatureType.NUMERIC:
            return feature_type.is_numeric()
        if self == FeatureType.CATEGORICAL:
            return feature_type.is_categorical()
        if self == FeatureType.NUMERIC_OR_ORDINAL:
            return feature_type.is_numeric_or_ordinal()
        return False

    def match_or_supertype_col(self, df_col):
        feature_type = FeatureType.get_feature_type(df_col)
        return self.match_or_supertype(feature_type)

    def match_or_supertype_df(self, df):
        for col in df:
            if not self.match_or_supertype_col(col):
                return False
        return True

    def __str__(self):
        if self == FeatureType.ANY:
            return 'any'
        if self == FeatureType.NUMERIC:
            return 'numeric'
        if self == FeatureType.NUMERIC_DISCRETE:
            return 'numeric discrete'
        if self == FeatureType.NUMERIC_CONTINUOUS:
            return 'numeric continuous'
        if self == FeatureType.CATEGORICAL:
            return 'categorical'
        if self == FeatureType.CATEGORICAL_ORDINAL:
            return 'categorical ordinal'
        if self == FeatureType.CATEGORICAL_NOMINAL:
            return 'categorical nominal'
        if self == FeatureType.NUMERIC_OR_ORDINAL:
            return 'numeric or ordinal'
        return 'other'

    def __repr__(self):
        return self.__str__()
