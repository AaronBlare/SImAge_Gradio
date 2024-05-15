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
from scipy import stats

import warnings

warnings.filterwarnings("ignore",
                        ".*will save all targets and predictions in the buffer. For large datasets, this may lead to large memory footprint.*")
warnings.filterwarnings("ignore", ".*is non-interactive, and thus cannot be shown*")

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

out_dir = f"{root_dir}/out"
if not os.path.exists(out_dir):
    os.makedirs(out_dir)


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
    if input.endswith('xlsx'):
        df = pd.read_excel(input, index_col=0)
    elif input.endswith('csv'):
        df = pd.read_csv(input, index_col=0)
    else:
        raise gr.Error(f"Unknown file type!")
    if "Age" not in df.columns:
        raise gr.Error("No 'Age' column in the input file!")
    missed_features = [feature for feature in feats if feature not in df.columns]
    if len(missed_features) > 0:
        raise gr.Error(f"No {', '.join(missed_features)} column(s) in the input file!")
    try:
        df = df.loc[:, feats + ['Age']]
    except ValueError:
        raise gr.Error(f"Non-numeric value in 'Age' column!")
    df = df.astype({'Age': 'float'})
    for feat in feats:
        try:
            df = df.astype({feat: 'float'})
        except ValueError:
            raise gr.Error(f"Non-numeric value in '{feat}' column!")

    df['SImAge'] = model(torch.from_numpy(df.loc[:, feats].values)).cpu().detach().numpy().ravel()
    df['SImAge acceleration'] = df['SImAge'] - df['Age']
    df.to_excel(f'{root_dir}/out/df.xlsx')

    df_res = df[['SImAge']]
    df_res.to_excel(f'{root_dir}/out/result.xlsx')

    if len(df) > 1:
        mae = mean_absolute_error(df['Age'].values, df['SImAge'].values)
        rho = pearsonr(df['Age'].values, df['SImAge'].values).statistic

    plt.close('all')

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

    plt.close('all')

    if len(df) > 1:
        sns.set_theme(style='whitegrid')
        fig, ax = plt.subplots(figsize=(2, 4))
        sns.violinplot(
            data=df,
            y='SImAge acceleration',
            density_norm='width',
            color='blue',
            saturation=0.75,
        )
        plt.savefig(f'{root_dir}/out/violin.svg', bbox_inches='tight')
        plt.close('all')

    shap.summary_plot(
        shap_values=shap_values_train.values,
        features=values_train.values,
        feature_names=feats,
        max_display=len(feats),
        plot_type="violin",
    )
    plt.savefig(f'{root_dir}/out/shap_beeswarm.svg', bbox_inches='tight')

    plt.close('all')

    if len(df) > 1:
        return_metrics = gr.update(value=f'MAE: {round(mae, 3)}\nPearson Rho: {round(rho, 3)}', visible=True)
        return_gallery = gr.update(value=[(f'{root_dir}/out/scatter.svg', 'Scatter'),
                                          (f'{root_dir}/out/violin.svg', 'Violin'),
                                          (f'{root_dir}/out/shap_beeswarm.svg', 'SHAP Beeswarm')], visible=True)
    else:
        return_metrics = gr.update(value=f'Only one sample.\nNo metrics can be calculated.', visible=True)
        return_gallery = gr.update(value=[(f'{root_dir}/out/scatter.svg', 'Scatter'),
                                          (f'{root_dir}/out/shap_beeswarm.svg', 'SHAP Beeswarm')], visible=True)

    return [return_metrics,
            gr.update(value=f'{root_dir}/out/result.xlsx', visible=True),
            return_gallery,
            gr.update(visible=True), gr.update(visible=True),
            gr.update(choices=list(df.index.values), value=list(df.index.values)[0], interactive=True, visible=True),
            gr.update(visible=True), gr.update(visible=True), gr.update(visible=True), gr.update(visible=True)]


def explain(input):
    df = pd.read_excel(f'{root_dir}/out/df.xlsx', index_col=0)

    trgt_id = input
    shap_values_trgt = explainer.shap_values(df.loc[trgt_id, feats].values)
    base_value = explainer.expected_value[0]

    age = df.loc[trgt_id, ['Age']].values[0]
    simage = df.loc[trgt_id, ['SImAge']].values[0]

    order = np.argsort(-np.abs(shap_values_trgt))
    locally_ordered_feats = [feats[i] for i in order]

    plt.close('all')

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

    plt.close('all')

    if len(df) > 1:

        age_window = 5
        trgt_age = df.at[trgt_id, 'Age']
        trgt_simage = df.at[trgt_id, 'SImAge']
        trgt_simage_acc = df.at[trgt_id, 'SImAge acceleration']
        ids_near = df.index[(df['Age'] >= trgt_age - age_window) & (df['Age'] < trgt_age + age_window)]
        trgt_simage_acc_prctl = stats.percentileofscore(df.loc[ids_near, 'SImAge acceleration'], trgt_simage_acc)

        sns.set(style='whitegrid', font_scale=1.5)
        fig, ax = plt.subplots(figsize=(10, 6))
        kdeplot = sns.kdeplot(
            data=df.loc[ids_near, :],
            x='SImAge acceleration',
            color='gray',
            linewidth=4,
            cut=0,
            ax=ax
        )
        kdeline = ax.lines[0]
        xs = kdeline.get_xdata()
        ys = kdeline.get_ydata()
        ax.fill_between(xs, 0, ys, where=(xs <= trgt_simage_acc), interpolate=True, facecolor='dodgerblue', alpha=0.7)
        ax.fill_between(xs, 0, ys, where=(xs >= trgt_simage_acc), interpolate=True, facecolor='crimson', alpha=0.7)
        ax.vlines(trgt_simage_acc, 0, np.interp(trgt_simage_acc, xs, ys), color='black', linewidth=6)
        ax.text(np.mean([min(xs), trgt_simage_acc]), 0.1 * max(ys), f"{trgt_simage_acc_prctl:0.1f}%",
                fontstyle="oblique",
                color="black", ha="center", va="center")
        ax.text(np.mean([max(xs), trgt_simage_acc]), 0.1 * max(ys), f"{100 - trgt_simage_acc_prctl:0.1f}%",
                fontstyle="oblique", color="black", ha="center", va="center")
        fig.savefig(f"{root_dir}/out/kde_aa_{trgt_id}.svg", bbox_inches='tight')
        plt.close(fig)

        sns.set(style='whitegrid', font_scale=0.7)
        n_rows = 2
        n_cols = 5
        fig_height = 4
        fig_width = 10
        fig, axs = plt.subplots(n_rows, n_cols, figsize=(fig_width, fig_height), gridspec_kw={}, sharey=False,
                                sharex=False)
        for feat_id, feat in enumerate(feats):
            row_id, col_id = divmod(feat_id, n_cols)
            kdeplot = sns.kdeplot(
                data=df.loc[ids_near, :],
                x=feat,
                color='gray',
                linewidth=1,
                cut=0,
                ax=axs[row_id, col_id]
            )
            kdeline = axs[row_id, col_id].lines[0]
            xs = kdeline.get_xdata()
            ys = kdeline.get_ydata()
            trgt_val = df.at[trgt_id, feat]
            trgt_prctl = stats.percentileofscore(df.loc[ids_near, feat], trgt_val)
            axs[row_id, col_id].fill_between(xs, 0, ys, where=(xs <= trgt_val), interpolate=True,
                                             facecolor='dodgerblue',
                                             alpha=0.7)
            axs[row_id, col_id].fill_between(xs, 0, ys, where=(xs >= trgt_val), interpolate=True, facecolor='crimson',
                                             alpha=0.7)
            axs[row_id, col_id].vlines(trgt_val, 0, np.interp(trgt_val, xs, ys), color='black', linewidth=1.5)
            axs[row_id, col_id].text(np.mean([min(xs), trgt_val]), 0.1 * max(ys), f"{trgt_prctl:0.1f}%",
                                     fontstyle="oblique",
                                     color="black", ha="center", va="center")
            axs[row_id, col_id].text(np.mean([max(xs), trgt_val]), 0.1 * max(ys), f"{100 - trgt_prctl:0.1f}%",
                                     fontstyle="oblique",
                                     color="black", ha="center", va="center")
            axs[row_id, col_id].ticklabel_format(style='scientific', scilimits=(-1, 1), axis='y', useOffset=True)
        fig.tight_layout()
        fig.savefig(f"{root_dir}/out/kde_feats_{trgt_id}.svg", bbox_inches='tight')
        plt.close(fig)

    if len(df) > 1:
        return_gallery = [(f'{root_dir}/out/waterfall_{trgt_id}.svg', 'Waterfall'),
                          (f'{root_dir}/out/kde_aa_{trgt_id}.svg', 'Age Acceleration KDE'),
                          (f'{root_dir}/out/kde_feats_{trgt_id}.svg', 'Features KDE')]
    else:
        return_gallery = [(f'{root_dir}/out/waterfall_{trgt_id}.svg', 'Waterfall')]

    return [f'Real age: {round(age, 3)}\nSImAge: {round(simage, 3)}',
            f'{locally_ordered_feats[0]}\n{locally_ordered_feats[1]}\n{locally_ordered_feats[2]}',
            return_gallery]


def clear():
    return (gr.update(interactive=False),
            gr.update(value=None, visible=False),
            gr.update(value=None, visible=False),
            gr.update(value=None, visible=False),
            gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False),
            gr.update(value=None, visible=False), gr.update(value=None, visible=False),
            gr.update(value=None, visible=False))


def check_size(input):
    curr_file_size = os.path.getsize(input)
    if curr_file_size > 1024 * 1024:
        raise gr.Error(f"File exceeds 1 MB limit!")
    else:
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
            input_file = gr.File(label='Input file', file_count='single', file_types=['.xlsx', 'csv'])
            submit_button = gr.Button("Submit data", variant="primary", interactive=False)
        with gr.Column():
            with gr.Row():
                output_text = gr.Text(label='Main metrics', visible=False)
                output_file = gr.File(label='Result file', file_types=['.xlsx'], interactive=False, visible=False)
            with gr.Row():
                gallery = gr.Gallery(label='Figures Gallery', object_fit='cover', columns=2, rows=2, visible=False)
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
                    shap_gallery = gr.Gallery(label='Local Explainability Gallery', object_fit='cover', columns=2,
                                              rows=2, visible=False)
    submit_button.click(fn=predict,
                        inputs=[input_file],
                        outputs=[output_text, output_file, gallery, title_shap, text_shap, input_shap, shap_button,
                                 shap_local,
                                 shap_cyto, shap_gallery]
                        )
    shap_button.click(fn=explain,
                      inputs=[input_shap],
                      outputs=[shap_local, shap_cyto, shap_gallery]
                      )
    input_file.clear(fn=clear,
                     inputs=[],
                     outputs=[submit_button, output_text, output_file, gallery,
                              title_shap, text_shap, input_shap, shap_button, shap_local, shap_cyto, shap_gallery])
    input_file.upload(fn=check_size,
                      inputs=[input_file],
                      outputs=[submit_button])
    gr.Markdown(
        """
        Reference:
        
        Kalyakulina, A., Yusipov, I., Kondakova, E., Bacalini, M. G., Franceschi, C., Vedunova, M., & Ivanchenko, M. (2023). [Small immunological clocks identified by deep learning and gradient boosting](https://www.frontiersin.org/journals/immunology/articles/10.3389/fimmu.2023.1177611/full). Frontiers in Immunology, 14, 1177611.
        """
    )
app.launch()
