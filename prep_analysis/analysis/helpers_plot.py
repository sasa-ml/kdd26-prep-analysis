import plotly.express as px
from husl import husl_to_rgb


def husl_palette(n, h=2, s=70, li=70):
    colors = []
    for i in range(n):
        starting_hue = h
        hue = (i + starting_hue) * 360 / n
        saturation = s
        lightness = li
        rgb = husl_to_rgb(hue, saturation, lightness)
        hex_color = '#%02x%02x%02x' % tuple(int(v * 255) for v in rgb)
        colors.append(hex_color)
    return colors


category_orders = {
    'i_dataset': list(range(51)),
    'pipeline': [
        '1_No_prep',
        '2_Basic',
        '3_Tree',
        '4_Tree_imb',
        '5_NN',
        '6_NN_imb',
        '7_prep',
    ],
    'model': [
        'c_RandomForest',
        'd_ExtraTrees',
        'e_XGBoost',
        'f_LightGBM',
        'g_CatBoost',
        'a_LinearModel',
        'b_KNeighbors',
        'h_NeuralNetFastAI',
        'i_TabM',
        'j_TabDPT',
        'k_RealTabPFN-v2.5',
    ],
    'acc_metric': [
        'accuracy',
        'balanced_accuracy',
    ],
}
pipeline_colors = px.colors.qualitative.Plotly
# pipeline_colors = husl_palette(len(category_orders['pipeline']))
pipeline_color_map = {
    p: pipeline_colors[i] for i, p in enumerate(category_orders['pipeline'])
}
# model_colors = px.colors.qualitative.Bold  # Prism, Bold
# model_colors = px.colors.sample_colorscale(
#    px.colors.sequential.Turbo_r,
#    # px.colors.diverging.Temps,
#    len(category_orders['model']),
# )
model_colors = husl_palette(len(category_orders['model']))
model_color_map = {
    p: model_colors[i] for i, p in enumerate(category_orders['model'])
}
model_error_colors = husl_palette(len(category_orders['model']), li=50)
model_error_color_map = {
    p: model_error_colors[i] for i, p in enumerate(category_orders['model'])
}
pattern_shapes = [
    '',
    '/',
    '\\',
    '|',
    '-',
    '+',
    'x',
    '.',
]
pipeline_pattern_map = {
    '1_No_prep': '',
    '2_Basic': '.',
    '3_Tree': '-',
    '4_Tree_imb': '+',
    '5_NN': '/',
    '6_NN_imb': 'x',
}
acc_metric_pattern_map = {
    'accuracy': '',
    'balanced_accuracy': '|',
}
symbols = [
    'circle',
    'square',
    'diamond',
    'cross',
    'x',
    'triangle-up',
    'triangle-down',
    'triangle-left',
    'triangle-right',
    'triangle-ne',
    'triangle-se',
    'triangle-sw',
    'triangle-nw',
    'pentagon',
    'hexagon',
    'hexagon2',
    'octagon',
    'star',
    'hexagram',
    'star-triangle-up',
    'star-triangle-down',
    'star-square',
    'star-diamond',
    'diamond-tall',
    'diamond-wide',
    'hourglass',
    'bowtie',
    'circle-cross',
    'circle-x',
    'square-cross',
    'square-x',
    'diamond-cross',
    'diamond-x',
    'cross-thin',
    'x-thin',
    'asterisk',
    'hash',
    'y-up',
    'y-down',
    'y-left',
    'y-right',
    'line-ew',
    'line-ns',
    'line-ne',
    'line-nw',
    'arrow-up',
    'arrow-down',
    'arrow-left',
    'arrow-right',
    'arrow-bar-up',
    'arrow-bar-down',
    'arrow-bar-left',
    'arrow-bar-right',
]
pipeline_symbol_map = {
    '1_No_prep': 'x',
    '2_Basic': 'circle',
    '3_Tree': 'square',
    '4_Tree_imb': 'diamond',
    '5_NN': 'pentagon',
    '6_NN_imb': 'star',
}
