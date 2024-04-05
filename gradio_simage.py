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

    df_res = df[['SImAge']]
    df_res.to_excel(f'{root_dir}/out/result.xlsx')

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
    plt.savefig(f'{root_dir}/out/shap_beeswarm.svg', bbox_inches='tight')

    return [f'MAE: {round(mae, 3)}\nPearson Rho: {round(rho, 3)}',
            f'{root_dir}/out/result.xlsx',
            [(f'{root_dir}/out/scatter.svg', 'Scatter'), (f'{root_dir}/out/violin.svg', 'Violin'),
             (f'{root_dir}/out/shap_beeswarm.svg', 'SHAP Beeswarm')],
            gr.update(visible=True), gr.update(visible=True),
            gr.update(choices=list(df.index.values), value=list(df.index.values)[0], interactive=True, visible=True),
            gr.update(visible=True), gr.update(visible=True), gr.update(visible=True), gr.update(visible=True)]


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

    order = np.argsort(-np.abs(shap_values_trgt))
    locally_ordered_feats = [feats[i] for i in order]

    return [f'Real age: {round(age, 3)}\nSImAge: {round(simage, 3)}',
            f'{locally_ordered_feats[0]}\n{locally_ordered_feats[1]}\n{locally_ordered_feats[2]}',
            f'{root_dir}/out/waterfall_{trgt_id}.svg']


def clear():
    return gr.update(interactive=False), gr.update(value=None), gr.update(value=None), gr.update(value=None), gr.update(
        visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(
        visible=False), gr.update(visible=False)


def active():
    return gr.update(interactive=True)


css = """
h2 {
    text-align: center;
    display:block;
}
"""

with gr.Blocks(css=css, theme=gr.themes.Soft(), title='SImAge') as app:
    gr.Markdown(
        """
        <h2>Calculate your immunological age using SImAge model</h2>
        """
    )
    with gr.Row():
        with gr.Column():
            gr.Markdown(
                """
                ### Submit immunology data
                The file should contain chronological age ("Age" column) and immunology data for the following 10 cytokines:

                CXCL9, CCL22, IL6, PDGFB, CD40LG, IL27, VEGFA, CSF1, PDGFA, CXCL10
                """
            )
            input_file = gr.File(label='Input file', file_count='single', file_types=['.xlsx', '.csv'])
            submit_button = gr.Button("Submit data", variant="primary", interactive=False)
        with gr.Column():
            with gr.Row():
                output_text = gr.Text(label='Main metrics')
                output_file = gr.File(label='Result file', file_types=['.xlsx'], interactive=False)
            with gr.Row():
                gallery = gr.Gallery(label='Figures Gallery', object_fit='cover', columns=2, rows=2)
    title_shap = gr.Markdown(
        """
        <h2>Local explainability</h2>
        """
        , visible=False)
    with gr.Row():
        with gr.Column():
            text_shap = gr.Markdown(
                """
                Select a record to get an explanation of the SImAge prediction:
                """
                , visible=False)
            input_shap = gr.Dropdown(label='Choose a sample', visible=False)
            shap_button = gr.Button("Get explanation", variant="primary", visible=False)
        with gr.Column():
            with gr.Row():
                with gr.Column(scale=1):
                    shap_local = gr.Text(label='Sample info', visible=False)
                    shap_cyto = gr.Text(label='Most important cytokines', visible=False)
                with gr.Column(scale=3):
                    shap_waterfall = gr.Image(show_label=False, visible=False)
    submit_button.click(fn=predict,
                        inputs=[input_file],
                        outputs=[output_text, output_file, gallery, title_shap, text_shap, input_shap, shap_button, shap_local,
                                 shap_cyto, shap_waterfall]
                        )
    shap_button.click(fn=explain,
                      inputs=[input_shap],
                      outputs=[shap_local, shap_cyto, shap_waterfall]
                      )
    input_file.clear(fn=clear,
                     inputs=[],
                     outputs=[submit_button, output_text, output_file, gallery,
                              title_shap, text_shap, input_shap, shap_button, shap_local, shap_cyto, shap_waterfall])
    input_file.upload(fn=active,
                      inputs=[],
                      outputs=[submit_button])
    gr.Markdown(
        """
        Reference:
        
        Kalyakulina, A., Yusipov, I., Kondakova, E., Bacalini, M. G., Franceschi, C., Vedunova, M., & Ivanchenko, M. (2023). [Small immunological clocks identified by deep learning and gradient boosting](https://www.frontiersin.org/journals/immunology/articles/10.3389/fimmu.2023.1177611/full). Frontiers in Immunology, 14, 1177611.
        """
    )
app.launch()
