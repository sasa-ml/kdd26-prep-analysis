from auto_prepper.core.optimizer import Optimizer
from auto_prepper.core.pipeline import Pipeline
from auto_prepper.core.pool import TransformationPool
from auto_prepper.pipelines.presets import PresetPipelineGenerator


class AdaptablePresetPipeline(Optimizer):

    def __init__(self, ds, eval_func=None, preset_type=None):
        super().__init__(ds, eval_func)
        self._preset_type = preset_type if preset_type else '1_No_prep'

    def _preset_pipeline(self):
        if self._preset_type == '1_No_prep':
            return PresetPipelineGenerator.generate_no_prep()
        if self._preset_type == '2_Basic':
            return PresetPipelineGenerator.generate_basic()
        if self._preset_type == '3_Tree':
            return PresetPipelineGenerator.generate_tree()
        if self._preset_type == '4_Tree_imb':
            return PresetPipelineGenerator.generate_tree_imb()
        if self._preset_type == '5_NN':
            return PresetPipelineGenerator.generate_nn()
        if self._preset_type == '6_NN_imb':
            return PresetPipelineGenerator.generate_nn_imb()
        return None

    def _get_flattened_transformation_pool(self, ds):
        t_pool = TransformationPool(ds).pool
        flattened_t_pool = {
            t.get_class_name()
            for t_category in t_pool
            for t in t_pool[t_category].keys()
        }
        return flattened_t_pool

    def _adapt_pipeline(self, pipeline):
        adapted_pipeline = Pipeline()
        transformation_list = pipeline.transformations
        ds = self._ds.copy()
        flattened_t_pool = self._get_flattened_transformation_pool(ds)
        while transformation_list:
            transformation = transformation_list.pop(0)
            if transformation.class_name in flattened_t_pool:
                if (
                    hasattr(transformation, '_hp_n_components')
                    and transformation.hyperparameters['n_components']
                    >= ds.df.width - 1
                ):
                    print(
                        f'Warning: {transformation} n_components '
                        + f'({transformation.hyperparameters['n_components']}'
                        + f') >= n_features ({ds.df.width - 1}). '
                        + 'Skipping transformation.'
                    )
                    continue
                ds = transformation.fit_transform(ds)
                adapted_pipeline.add_as_is(transformation)
                flattened_t_pool = self._get_flattened_transformation_pool(ds)
        adapted_pipeline.set_fitted(True)
        return adapted_pipeline

    def optimize_pipeline(self):
        pipeline = self._preset_pipeline()
        pipeline = self._adapt_pipeline(pipeline)
        return pipeline
