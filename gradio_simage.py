import shap
import pandas as pd
import os
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt
import torch
import numpy as np
import pickle
from sklearn.metrics import mean_absolute_error
from scipy.stats import pearsonr
from models.tabular.widedeep.ft_transformer import WDFTTransformerModel
import gradio as gr

root_dir = Path(os.getcwd())

fn_model = f"{root_dir}/data/model.ckpt"
model = WDFTTransformerModel.load_from_checkpoint(checkpoint_path=fn_model)
model.eval()
model.freeze()

feats = [
    'CXCL9',
    'CCL22',
    'IL6',
    'PDGFB',
    'CD40LG',
    'IL27',
    'VEGFA',
    'CSF1',
    'PDGFA',
    'CXCL10'
]

fn_shap = f"{root_dir}/data/shap.pickle"


def predict_func(x):
    batch = {
        'all': torch.from_numpy(np.float32(x)),
        'continuous': torch.from_numpy(np.float32(x)),
        'categorical': torch.from_numpy(np.int32(x[:, []])),
    }
    return model(batch).cpu().detach().numpy()


with open(fn_shap, 'rb') as handle:
    shap_dict = pickle.load(handle)
values_train = shap_dict['values_train']
shap_values_train = shap_dict['shap_values_train']
explainer = shap_dict['explainer']


def predict(input):
    df = pd.read_excel(input, index_col=0)
    df = df.loc[:, feats + ['Age']]

    df['SImAge'] = model(torch.from_numpy(df.loc[:, feats].values)).cpu().detach().numpy().ravel()
    df['SImAge acceleration'] = df['SImAge'] - df['Age']
    df.to_excel(f'{root_dir}/out/df.xlsx')

    df_res = df[['SImAge acceleration']]
    df_res.to_excel(f'{root_dir}/out/output.xlsx')

    mae = mean_absolute_error(df['Age'].values, df['SImAge'].values)
    rho = pearsonr(df['Age'].values, df['SImAge'].values).statistic

    sns.set_theme(style='whitegrid')
    fig, ax = plt.subplots(figsize=(4, 4))
    scatter = sns.scatterplot(
        data=df,
        x="Age",
        y="SImAge",
        linewidth=0.1,
        alpha=0.75,
        edgecolor="k",
        s=40,
        color='blue',
        ax=ax
    )
    bisect = sns.lineplot(
        x=[0, 120],
        y=[0, 120],
        linestyle='--',
        color='black',
        linewidth=1.0,
        ax=ax
    )
    ax.set_xlim(0, 120)
    ax.set_ylim(0, 120)
    plt.savefig(f'{root_dir}/out/scatter.svg', bbox_inches='tight')

    sns.set_theme(style='whitegrid')
    fig, ax = plt.subplots(figsize=(2, 4))
    sns.violinplot(
        data=df,
        y='SImAge acceleration',
        scale='width',
        color='blue',
        saturation=0.75,
    )
    plt.savefig(f'{root_dir}/out/violin.svg', bbox_inches='tight')

    shap.summary_plot(
        shap_values=shap_values_train.values,
        features=values_train.values,
        feature_names=feats,
        max_display=len(feats),
        plot_type="violin",
    )
    plt.savefig(f'{root_dir}/out/shap.svg', bbox_inches='tight')

    return [f'MAE: {round(mae, 3)}, Pearson Rho: {round(rho, 3)}',
            f'{root_dir}/out/output.xlsx',
            f'{root_dir}/out/scatter.svg', f'{root_dir}/out/violin.svg', f'{root_dir}/out/shap.svg',
            gr.update(choices=list(df.index.values), value=list(df.index.values)[0], interactive=True, visible=True),
            gr.update(visible=True), gr.update(visible=True), gr.update(visible=True)]


def explain(input):
    df = pd.read_excel(f'{root_dir}/out/df.xlsx', index_col=0)

    trgt_id = input
    shap_values_trgt = explainer.shap_values(df.loc[trgt_id, feats].values)
    base_value = explainer.expected_value[0]

    shap.plots.waterfall(
        shap.Explanation(
            values=shap_values_trgt,
            base_values=base_value,
            data=df.loc[trgt_id, feats].values,
            feature_names=feats
        ),
        max_display=len(feats),
        show=True,
    )
    plt.savefig(f'{root_dir}/out/waterfall_{trgt_id}.svg', bbox_inches='tight')

    age = df.loc[trgt_id, ['Age']].values[0]
    simage = df.loc[trgt_id, ['SImAge']].values[0]

    return [f'Real age: {round(age, 3)}, SImAge: {round(simage, 3)}',
            f'{root_dir}/out/waterfall_{trgt_id}.svg']


with gr.Blocks(theme=gr.themes.Soft()) as app:
    with gr.Row():
        with gr.Column():
            input_file = gr.File(label='Input file', file_count='single', file_types=['.xlsx', '.csv'])
            submit_button = gr.Button("Submit data", variant="primary")
            input_shap = gr.Dropdown(label='Choose a sample', visible=False)
            shap_button = gr.Button("Get explanation", variant="primary", visible=False)
        with gr.Column():
            output_text = gr.Text(label='Main metrics')
            output_file = gr.File(label='Output file', file_types=['.xlsx'], interactive=False)
            with gr.Row():
                scatter_image = gr.Image(label='Scatter')
                violin_image = gr.Image(label='Violin')
                shap_image = gr.Image(label='SHAP')
            shap_local = gr.Text(label='Main metrics', visible=False)
            shap_waterfall = gr.Image(label='Waterfall', visible=False)
    submit_button.click(fn=predict,
                        inputs=[input_file],
                        outputs=[output_text, output_file, scatter_image, violin_image, shap_image, input_shap, shap_button, shap_local, shap_waterfall]
                        )
    shap_button.click(fn=explain,
                      inputs=[input_shap],
                      outputs=[shap_local, shap_waterfall]
                      )
app.launch(share=True)