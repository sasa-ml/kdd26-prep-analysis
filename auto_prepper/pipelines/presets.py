from auto_prepper.core.pipeline import Pipeline
from auto_prepper.transformations.dimensionality_reduction import (
    DimReduceTruncatedSVD,
)
from auto_prepper.transformations.duplicates import DropDuplicates
from auto_prepper.transformations.encoding import (
    EncodeHash,
    EncodeInt,
    EncodeIntTarget,
    EncodeOneHotOrHash,
)
from auto_prepper.transformations.feature_generation import TemporalExpand
from auto_prepper.transformations.feature_selection import (
    DropInvariantColumns,
    SelectUnivariateMutualInformationCls,
    SelectUnivariateMutualInformationReg,
)
from auto_prepper.transformations.missing_values import (
    DropNoneColumns,
    DropNoneRows,
    FillNaNWithNone,
    FillNoneKNNOrMean,
    FillNoneMean,
    FillNoneMode,
)
from auto_prepper.transformations.oversampling import (
    OversampleRandom,
    OversampleSMOTEOrRandom,
)
from auto_prepper.transformations.scaling import (
    Normalize,
    QuantileTransform,
    Standardize,
)
from auto_prepper.transformations.type_handling import (
    BooleanToCategorical,
    DictUnnest,
    DropUnsupportedTypes,
    ListExplode,
    StrDecode,
    StrToTemporal,
    TemporalToInt,
)


class PresetPipelineGenerator:

    @classmethod
    def generate_no_prep(cls):
        pipeline = Pipeline()
        return pipeline

    @classmethod
    def generate_basic(cls):
        pipeline = Pipeline()
        pipeline.extend(cls.generate_type_standardization())
        pipeline.extend(cls.generate_cleanup())
        pipeline.add(FillNoneMean())
        pipeline.add(FillNoneMode())
        pipeline.add(Normalize())
        pipeline.add(EncodeIntTarget())
        pipeline.add(EncodeHash())
        # pipeline.add(SelectUnivariateMutualInformationReg(n_features=100))
        # pipeline.add(SelectUnivariateMutualInformationCls(n_features=100))
        return pipeline

    @classmethod
    def generate_tree(cls):
        pipeline = Pipeline()
        pipeline.extend(cls.generate_type_standardization())
        pipeline.extend(cls.generate_structures())
        pipeline.extend(cls.generate_temporal())
        pipeline.extend(cls.generate_cleanup())
        pipeline.add(FillNoneMean())
        pipeline.add(FillNoneMode())
        pipeline.add(DropDuplicates())
        pipeline.add(EncodeIntTarget())
        pipeline.add(EncodeInt())
        # pipeline.add(SelectUnivariateMutualInformationReg(n_features=100))
        # pipeline.add(SelectUnivariateMutualInformationCls(n_features=100))
        return pipeline

    @classmethod
    def generate_tree_imb(cls):
        pipeline = Pipeline()
        pipeline.extend(cls.generate_type_standardization())
        pipeline.extend(cls.generate_structures())
        pipeline.extend(cls.generate_temporal())
        pipeline.extend(cls.generate_cleanup())
        pipeline.add(FillNoneMean())
        pipeline.add(FillNoneMode())
        pipeline.add(DropDuplicates())
        pipeline.add(QuantileTransform(n_quantiles=1000))
        pipeline.add(EncodeIntTarget())
        pipeline.add(EncodeInt())
        pipeline.add(OversampleRandom())
        # pipeline.add(SelectUnivariateMutualInformationReg(n_features=100))
        # pipeline.add(SelectUnivariateMutualInformationCls(n_features=100))
        return pipeline

    @classmethod
    def generate_nn(cls):
        pipeline = Pipeline()
        pipeline.extend(cls.generate_type_standardization())
        pipeline.extend(cls.generate_structures())
        pipeline.extend(cls.generate_temporal())
        pipeline.extend(cls.generate_cleanup())
        pipeline.add(FillNoneKNNOrMean(n_neighbors=2))
        pipeline.add(FillNoneMode())
        pipeline.add(DropDuplicates())
        pipeline.add(Standardize())
        pipeline.add(EncodeIntTarget())
        pipeline.add(EncodeOneHotOrHash(max_values_1hot=20))
        pipeline.add(DimReduceTruncatedSVD(n_components=2000))
        # pipeline.add(SelectUnivariateMutualInformationReg(n_features=100))
        # pipeline.add(SelectUnivariateMutualInformationCls(n_features=100))
        return pipeline

    @classmethod
    def generate_nn_imb(cls):
        pipeline = Pipeline()
        pipeline.extend(cls.generate_type_standardization())
        pipeline.extend(cls.generate_structures())
        pipeline.extend(cls.generate_temporal())
        pipeline.extend(cls.generate_cleanup())
        pipeline.add(FillNoneKNNOrMean(n_neighbors=2))
        pipeline.add(FillNoneMode())
        pipeline.add(DropDuplicates())
        pipeline.add(QuantileTransform(n_quantiles=1000))
        pipeline.add(EncodeIntTarget())
        pipeline.add(EncodeOneHotOrHash(max_values_1hot=20))
        pipeline.add(DimReduceTruncatedSVD(n_components=2000))
        # pipeline.add(SelectUnivariateMutualInformationReg(n_features=100))
        # pipeline.add(SelectUnivariateMutualInformationCls(n_features=100))
        pipeline.add(OversampleSMOTEOrRandom(n_neighbors=2))
        return pipeline

    @classmethod
    def generate_type_standardization(cls):
        return Pipeline(
            BooleanToCategorical(),
        )

    @classmethod
    def generate_structures(cls):
        return Pipeline(
            StrDecode(),
            ListExplode(),
            DictUnnest(),
        )

    @classmethod
    def generate_temporal(cls):
        return Pipeline(
            StrToTemporal(),
            TemporalExpand(),
            TemporalToInt(),
        )

    @classmethod
    def generate_cleanup(cls):
        return Pipeline(
            DropUnsupportedTypes(),
            FillNaNWithNone(),
            DropNoneColumns(),
            DropNoneRows(),
            DropInvariantColumns(),
        )
