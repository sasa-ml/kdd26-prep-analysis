from auto_prepper.core.profiling import Profile
from auto_prepper.core.transformation import (
    TransformationNoColParam,
    TransformationTargetOnly,
)
from auto_prepper.transformations.dimensionality_reduction import (
    DimensionalityReduction,
)
from auto_prepper.transformations.duplicates import Duplicates
from auto_prepper.transformations.encoding import Encoding
from auto_prepper.transformations.feature_generation import FeatureGeneration
from auto_prepper.transformations.feature_selection import FeatureSelection
from auto_prepper.transformations.missing_values import MissingValues
from auto_prepper.transformations.outliers import Outliers
from auto_prepper.transformations.oversampling import Oversampling
from auto_prepper.transformations.scaling import Scaling
from auto_prepper.transformations.type_handling import DataTypes
from auto_prepper.transformations.undersampling import Undersampling
from auto_prepper.utils.feature_type import FeatureType
from auto_prepper.utils.helpers import (
    get_class_subclasses,
    get_class_superclasses,
)


class TransformationPool:
    _transformation_categories = [
        DataTypes,
        MissingValues,
        Duplicates,
        Outliers,
        Encoding,
        Oversampling,
        Undersampling,
        Scaling,
        FeatureGeneration,
        FeatureSelection,
        DimensionalityReduction,
    ]

    def __init__(self, ds, profile=None):
        self._ds = ds
        if profile:
            self._profile = profile.result
        else:
            self._profile = Profile(self._ds).profile
        self._pool = {}
        self._fill_transformation_pool()

    @property
    def pool(self):
        return self._pool

    def _get_transformations_in_category(self, transformation_category):
        transformations = get_class_subclasses(transformation_category)
        return transformations

    def _init_pool(self):
        for cat in self._transformation_categories:
            self._pool[cat] = {}
            transformations = self._get_transformations_in_category(cat)
            for t in transformations:
                self._pool[cat][t] = self._ds.df.columns

    def _filter_req_target_column(self):
        if self._ds.target_column:
            return
        to_remove = []
        for cat in self._pool:
            for t in self._pool[cat]:
                if t.get_req_target_column():
                    to_remove.append((cat, t))
        for cat, t in to_remove:
            del self._pool[cat][t]
        self._clean_pool()

    def _filter_feature_type(self):
        to_remove = []
        for cat in self._pool:
            for t in self._pool[cat]:
                if self._pool[cat][t] is None:
                    continue
                t_feature_type = t.get_feature_type()
                if t_feature_type != FeatureType.ANY:
                    cols = [
                        col
                        for col in self._pool[cat][t]
                        if t_feature_type.match_or_supertype(
                            self._profile['feature_types'][col]
                        )
                    ]
                    if cols:
                        self._pool[cat][t] = cols
                    else:
                        to_remove.append((cat, t))
        for cat, t in to_remove:
            del self._pool[cat][t]
        self._clean_pool()

    def _filter_target_type(self):
        to_remove = []
        for cat in self._pool:
            for t in self._pool[cat]:
                t_target_type = t.get_target_type()
                if t_target_type != FeatureType.ANY:
                    if not t_target_type.match_or_supertype(
                        self._profile['feature_types'][self._ds.target_column]
                    ):
                        to_remove.append((cat, t))
        for cat, t in to_remove:
            del self._pool[cat][t]
        self._clean_pool()

    def _filter_data_types(self):
        to_remove = []
        for cat in self._pool:
            for t in self._pool[cat]:
                if self._pool[cat][t] is None:
                    continue
                t_data_types = t.get_data_types()
                if t_data_types:
                    cols = [
                        col
                        for col in self._pool[cat][t]
                        if self._profile['data_types'][col] in t_data_types
                    ]
                    if cols:
                        self._pool[cat][t] = cols
                    else:
                        to_remove.append((cat, t))
        for cat, t in to_remove:
            del self._pool[cat][t]
        self._clean_pool()

    def _filter_str_subtypes(self):
        categories = [DataTypes]
        to_remove = []
        for cat in categories:
            if cat not in self._pool:
                continue
            for t in self._pool[cat]:
                if self._pool[cat][t] is None:
                    continue
                t_str_subtype = t.get_str_subtype()
                if t_str_subtype:
                    cols = [
                        col
                        for col in self._pool[cat][t]
                        if self._profile['str_subtypes'][col] == t_str_subtype
                    ]
                    if cols:
                        self._pool[cat][t] = cols
                    else:
                        to_remove.append((cat, t))
        for cat, t in to_remove:
            del self._pool[cat][t]
        self._clean_pool(categories)

    def _filter_duplicates(self):
        categories = [Duplicates]
        if self._profile['duplicates']['count'] <= 0:
            for cat in categories:
                if cat not in self._pool:
                    continue
                del self._pool[cat]
        self._clean_pool(categories)

    def _filter_feature_count(self):
        to_remove = []
        for cat in self._pool:
            for t in self._pool[cat]:
                if self._pool[cat][t] is None:
                    continue
                columns = self._pool[cat][t]
                n_features = len(columns)
                # TODO if target not always excluded
                # if (
                #    self._ds.target_column
                #    and self._ds.target_column in columns
                # ):
                #    n_features -= 1
                if n_features < t.get_min_features():
                    to_remove.append((cat, t))
        for cat, t in to_remove:
            del self._pool[cat][t]
        self._clean_pool()

    def _filter_outliers(self):
        categories = [Outliers]
        if not self._profile['summary']['cols_w_outliers']:
            for cat in categories:
                if cat not in self._pool:
                    continue
                del self._pool[cat]
        self._clean_pool(categories)

    def _filter_class_imbalance(self):
        categories = [Oversampling, Undersampling]
        to_remove = []
        if not self._profile['summary']['imbalanced_target']:
            for cat in categories:
                if cat not in self._pool:
                    continue
                to_remove.append((cat))
        for cat in to_remove:
            del self._pool[cat]
        self._clean_pool(categories)

    def _filter_exclude_target(self):
        categories_with_remove = set()
        for cat in self._pool:
            for t in self._pool[cat]:
                if (
                    t.get_exclude_target()
                    and self._pool[cat][t]
                    and self._ds.target_column in self._pool[cat][t]
                ):
                    self._pool[cat][t].remove(self._ds.target_column)
                    categories_with_remove.add(cat)
        self._clean_pool(list(categories_with_remove))

    def _filter_target_only(self):
        to_remove = []
        for cat in self._pool:
            for t in self._pool[cat]:
                superclasses = get_class_superclasses(t)
                if TransformationTargetOnly in superclasses:
                    if self._ds.target_column in self._pool[cat][t]:
                        self._pool[cat][t] = [self._ds.target_column]
                    else:
                        to_remove.append((cat, t))
        for cat, t in to_remove:
            del self._pool[cat][t]
        self._clean_pool()

    def _clean_pool(self, categories=None):
        if not categories:
            categories = list(self._pool.keys())
        for cat in categories:
            if cat not in self._pool:
                continue
            to_remove = []
            for t in self._pool[cat]:
                if self._pool[cat][t] == []:
                    to_remove.append((cat, t))
            for cat, t in to_remove:
                del self._pool[cat][t]
            if not self._pool[cat]:
                del self._pool[cat]

    def _clean_pool_nocolparam(self):
        for cat in self._pool:
            for t in self._pool[cat]:
                superclasses = get_class_superclasses(t)
                if TransformationNoColParam in superclasses:
                    self._pool[cat][t] = None

    def _fill_transformation_pool(self):
        self._init_pool()
        self._filter_req_target_column()
        self._filter_feature_type()
        self._filter_target_type()
        self._filter_data_types()
        self._filter_str_subtypes()
        self._filter_duplicates()
        self._filter_feature_count()
        self._filter_outliers()
        self._filter_class_imbalance()
        self._filter_exclude_target()
        self._filter_target_only()
        self._clean_pool_nocolparam()

    def __str__(self):
        s = ''
        for cat in self._pool:
            s += f'\n{cat.__name__}'
            for t in self._pool[cat]:
                s += f'\n\t{t.__name__}: {self._pool[cat][t]}'
        return s

    def __repr__(self):
        return self.__str__()
