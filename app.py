import os
import time
import threading
import streamlit as st
import torch
import json
import numpy as np
from PIL import Image
import pandas as pd
import plotly.graph_objects as go
import logging
logging.getLogger('streamlit').setLevel(logging.ERROR)

from dataset import FER2013Dataset, get_transforms, get_dataloaders
from models import ModelA_Small, ModelB_Medium, ModelC_Large
from train import train_model
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    auc
)
# ============ 页面配置 ============
st.set_page_config(
    page_title="人脸情感识别系统",
    page_icon="😊",
    layout="wide",
)

st.title("😊 人脸情感识别应用")
st.markdown("基于 FER2013 数据集，使用 PyTorch CNN 进行 7 种表情分类")

# ============ 侧边栏 ============
st.sidebar.header("📂 数据配置")
csv_path = st.sidebar.text_input(
    "FER2013 CSV 文件路径",
    value="data/fer2013.csv",
    help="下载 fer2013.csv 后放入 data/ 目录"
)

st.sidebar.header("⚙️ 训练参数")
batch_size = st.sidebar.selectbox("Batch Size", [32, 64, 128], index=1)
learning_rate = st.sidebar.selectbox("Learning Rate", [1e-3, 5e-4, 1e-4], index=0)
epochs = st.sidebar.slider("Max Epochs", 10, 60, 30)
patience = st.sidebar.slider("Early Stopping Patience", 3, 15, 7)

# 设备选择
device_str = st.sidebar.radio(
    "运行设备",
    ["CPU", "CUDA (GPU)"],
    index=0 if not torch.cuda.is_available() else 1,
    disabled=not torch.cuda.is_available()
)
device = torch.device('cuda' if device_str == "CUDA (GPU)" and torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)
    st.sidebar.success(f"检测到 GPU: **{gpu_name}**")
else:
    st.sidebar.warning("未检测到 CUDA GPU，将使用 CPU 训练（请检查 PyTorch 安装版本）")
st.sidebar.info(f"当前设备: **{device}**")

# ============ 主界面 Tabs ============
tab1, tab2, tab3, tab4 = st.tabs([
    "📤 图像识别",
    "🚀 模型训练",
    "📊 模型对比",
    "📋 训练日志"
])

# ============ Tab 1: 图像识别 ============
with tab1:
    st.header("上传图像进行表情识别")

    # 选择用于预测的模型
    pred_model_name = st.selectbox(
        "选择模型",
        ["Model A (Small CNN)", "Model B (Medium CNN)", "Model C (Large CNN)"],
        key="pred_model"
    )

    uploaded_file = st.file_uploader(
        "上传一张人脸图像",
        type=['jpg', 'jpeg', 'png', 'bmp'],
        help="建议上传正面人脸图像，系统将自动转换为灰度图"
    )

    if uploaded_file is not None:
        col1, col2 = st.columns(2)

        with col1:
            image = Image.open(uploaded_file).convert('L')  # 转为灰度
            st.image(image, caption="上传的图像（灰度处理后）", width='stretch')

        with col2:
            if st.button("🔍 开始识别"):
                with st.spinner("正在识别..."):
                    try:
                        if pred_model_name == "Model A (Small CNN)":
                            model = ModelA_Small()
                            ckpt_path = "checkpoints/ModelA_best.pt"
                        elif pred_model_name == "Model B (Medium CNN)":
                            model = ModelB_Medium()
                            ckpt_path = "checkpoints/ModelB_best.pt"
                        else:
                            model = ModelC_Large()
                            ckpt_path = "checkpoints/ModelC_best.pt"
                        if os.path.exists(ckpt_path):
                            model.load_state_dict(torch.load(ckpt_path, map_location=device))
                        model = model.to(device)
                        model.eval()
                        transform = get_transforms(augment=False)
                        img_tensor = transform(image).unsqueeze(0).to(device)
                        with torch.no_grad():
                            output = model(img_tensor)
                            probs = torch.softmax(output, dim=1).cpu().numpy()[0]

                        emotion_names = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
                        emotion_cn = ['😠 愤怒', '🤢 厌恶', '😨 恐惧', '😄 快乐', '😢 悲伤', '😲 惊讶', '😐 中性']
                        pred_idx = np.argmax(probs)
                        st.success(f"**识别结果**: {emotion_cn[pred_idx]} (置信度: {probs[pred_idx]:.2%})")
                        st.markdown("### 各类别概率分布")
                        fig = go.Figure(data=[
                            go.Bar(x=emotion_cn, y=probs, text=[f'{p:.2%}' for p in probs],
                                   textposition='auto', marker_color=[
                                    '#FF6B6B' if i == pred_idx else '#A0A0A0' for i in range(7)
                                ])
                        ])
                        fig.update_layout(
                            yaxis_title='Probability',
                            yaxis_range=[0, 1],
                            height=300,
                            margin=dict(l=10, r=10, t=10, b=10),
                        )
                        st.plotly_chart(fig, width='stretch')
                    except Exception as e:
                        st.error(f"识别失败: {str(e)}")
                        st.info("请确保已训练并保存对应模型权重")

# ============ Tab 2: 模型训练 ============
with tab2:
    st.header("模型训练控制面板")

    if not os.path.exists(csv_path):
        st.error(f"❌ 未找到数据集文件: `{csv_path}`。请先下载 FER2013 数据集并放入对应目录。")
        st.info("数据集下载方式请参考侧边栏说明。")
    else:
        st.success(f"✅ 已找到数据集: `{csv_path}`")

    # 模型选择
    model_choice = st.selectbox(
        "选择要训练的模型",
        ["Model A (Small CNN)", "Model B (Medium CNN)", "Model C (Large CNN)"]
    )

    # 模型信息卡片
    model_info = {
        "Model A (Small CNN)": {
            "layers": "2 层卷积 (5×5 大卷积核)",
            "filters": "32 → 64",
            "params": "~450K",
            "desc": "轻量级，适合快速实验。大卷积核带来较大的单层感受野，但非线性表达能力相对较弱。",
        },
        "Model B (Medium CNN)": {
            "layers": "3 层卷积 (3×3 小卷积核)",
            "filters": "64 → 128 → 256",
            "params": "~1.5M",
            "desc": "均衡选择。3层 3×3 卷积堆叠等效于更大的感受野，同时引入更多非线性。",
        },
        "Model C (Large CNN)": {
            "layers": "4 层卷积 (3×3 小卷积核)",
            "filters": "64 → 128 → 256 → 512",
            "params": "~3.2M",
            "desc": "深度模型。更多层数和滤波器带来更强特征提取能力，但训练时间更长，需要更多数据避免过拟合。",
        },
    }
    info = model_info[model_choice]
    st.info(f"**{model_choice}**: {info['desc']} | 卷积层数: {info['layers']} | 参数量: {info['params']}")

    # 训练进度占位符
    progress_bar = st.progress(0)
    status_text = st.empty()

    # 训练指标实时图表
    metrics_placeholder = st.empty()

    # 日志显示区域
    log_expander = st.expander("📋 训练日志", expanded=True)
    log_placeholder = log_expander.empty()

    # 混淆矩阵和 ROC 曲线
    col_cm, col_roc = st.columns(2)
    cm_placeholder = col_cm.empty()
    roc_placeholder = col_roc.empty()

    # 控制按钮
    col_btn1, col_btn2, col_btn3, col_btn4 = st.columns(4)

    if 'training_state' not in st.session_state:
        st.session_state.training_state = 'idle'  # idle / running / paused / stopped
        st.session_state.training_thread = None
        st.session_state.log_lines = []


    def on_start():
        st.session_state.training_state = 'running'


    def on_pause():
        st.session_state.training_state = 'paused'


    def on_continue():
        st.session_state.training_state = 'running'


    def on_stop():
        st.session_state.training_state = 'stopped'


    start_btn = col_btn1.button("▶️ 开始训练", on_click=on_start,
                                disabled=(st.session_state.training_state == 'running'))
    pause_btn = col_btn2.button("⏸️ 暂停", on_click=on_pause, disabled=(st.session_state.training_state != 'running'))
    continue_btn = col_btn3.button("▶️ 继续", on_click=on_continue,
                                   disabled=(st.session_state.training_state != 'paused'))
    stop_btn = col_btn4.button("⏹️ 停止", on_click=on_stop,
                               disabled=(st.session_state.training_state not in ['running', 'paused']))

    # 执行训练
    if st.session_state.training_state == 'running' and not os.path.exists(csv_path):
        st.error("请先配置正确的数据集路径！")
        st.session_state.training_state = 'idle'
    elif st.session_state.training_state == 'running':
        os.makedirs('checkpoints', exist_ok=True)

        # 初始化模型
        if model_choice == "Model A (Small CNN)":
            model = ModelA_Small()
            model_name = "ModelA"
        elif model_choice == "Model B (Medium CNN)":
            model = ModelB_Medium()
            model_name = "ModelB"
        else:
            model = ModelC_Large()
            model_name = "ModelC"

        model = model.to(device)

        # 加载数据
        with st.spinner("正在加载数据..."):
            train_loader, val_loader, test_loader = get_dataloaders(csv_path, batch_size=batch_size)
            status_text.text(
                f"数据加载完成 | 训练集: {len(train_loader.dataset)} 张, 验证集: {len(val_loader.dataset)} 张")

        # 训练循环
        all_logs = []
        train_losses, val_losses = [], []
        train_accs, val_accs, val_f1s = [], [], []

        for result in train_model(model, train_loader, val_loader, device,
                                  epochs=epochs, lr=learning_rate,
                                  patience=patience, model_name=model_name):
            # 检查暂停
            while st.session_state.training_state == 'paused':
                time.sleep(0.1)
            # 检查停止
            if st.session_state.training_state == 'stopped':
                status_text.text("⚠️ 训练已停止")
                break

            if result['type'] == 'log':
                all_logs.append(result['message'])
                log_placeholder.text('\n'.join(all_logs[-20:]))

            elif result['type'] == 'epoch_result':
                epoch = result['epoch']
                # 进度条
                progress_bar.progress(epoch / epochs)

                # 更新指标
                train_losses.append(result['train_loss'])
                val_losses.append(result['val_loss'])
                train_accs.append(result['train_acc'])
                val_accs.append(result['val_acc'])
                val_f1s.append(result['val_f1'])

                # 日志
                all_logs.append(result['log_message'])
                log_placeholder.text('\n'.join(all_logs[-20:]))

                # 状态
                status_text.text(f"Epoch {epoch}/{epochs} | Train Acc: {result['train_acc']:.4f} | "
                                 f"Val Acc: {result['val_acc']:.4f} | Best F1: {result['best_val_f1']:.4f}")

                # 绘制实时曲线
                fig = go.Figure()
                epochs_range = list(range(1, len(train_losses) + 1))
                fig.add_trace(go.Scatter(x=epochs_range, y=train_losses, mode='lines+markers', name='Train Loss'))
                fig.add_trace(go.Scatter(x=epochs_range, y=val_losses, mode='lines+markers', name='Val Loss'))
                fig.add_trace(go.Scatter(x=epochs_range, y=train_accs, mode='lines+markers', name='Train Acc'))
                fig.add_trace(go.Scatter(x=epochs_range, y=val_accs, mode='lines+markers', name='Val Acc'))
                fig.add_trace(go.Scatter(x=epochs_range, y=val_f1s, mode='lines+markers', name='Val F1'))
                fig.update_layout(height=350, margin=dict(l=10, r=10, t=30, b=10),
                                  title="训练指标实时曲线", hovermode='x unified')
                metrics_placeholder.plotly_chart(fig, width='stretch')

                # 混淆矩阵和 ROC
                cm_placeholder.image(result['cm_image'], caption=f"混淆矩阵 - Epoch {epoch}", width='stretch')
                roc_placeholder.image(result['roc_image'], caption=f"ROC 曲线 - Epoch {epoch}",
                                      width='stretch')

                if result.get('stopped_early'):
                    status_text.text(f"⏹️ 早停: Epoch {epoch}")
                    break

            elif result['type'] == 'complete':
                all_logs.append(result['message'])
                log_placeholder.text('\n'.join(all_logs[-20:]))
                status_text.text(result['message'])
                progress_bar.progress(1.0)
                best_epoch = result.get('best_epoch', '?') 
                cm_path = f'checkpoints/{model_name}_best_cm.png'
                roc_path = f'checkpoints/{model_name}_best_roc.png'
                if os.path.exists(cm_path) and os.path.exists(roc_path):
                    st.markdown("---")
                    st.subheader("🏆 最佳 Epoch 评估图")
                    col_best1, col_best2 = st.columns(2)
                    col_best1.image(cm_path, caption=f'最佳混淆矩阵 (Epoch {best_epoch})', width='stretch')
                    col_best2.image(roc_path, caption='最佳 ROC 曲线', width='stretch')

        # 重置状态
        st.session_state.training_state = 'idle'

        # 在测试集上评估
        if st.button("📊 在测试集上评估最佳模型", key="eval_test"):
            with st.spinner("正在评估..."):
                model.load_state_dict(torch.load(f'checkpoints/{model_name}_best.pt'))
                model.eval()
                all_preds, all_labels, all_probs = [], [], []
                with torch.no_grad():
                    for images, labels in test_loader:
                        images, labels = images.to(device), labels.to(device)
                        outputs = model(images)
                        probs = torch.softmax(outputs, dim=1)
                        _, predicted = torch.max(outputs, 1)
                        all_preds.extend(predicted.cpu().numpy())
                        all_labels.extend(labels.cpu().numpy())
                        all_probs.extend(probs.cpu().numpy())

                test_acc = accuracy_score(all_labels, all_preds)
                test_f1 = f1_score(all_labels, all_preds, average='macro')
                st.success(f"测试集准确率: **{test_acc:.4f}** | 测试集 Macro F1: **{test_f1:.4f}**")
                st.text('\n' + classification_report(all_labels, all_preds,
                                                     target_names=['Angry', 'Disgust', 'Fear', 'Happy', 'Sad',
                                                                   'Surprise', 'Neutral']))

# ============ Tab 3: 模型对比 ============
with tab3:
    st.header("三模型对比分析")

    st.markdown("""
    ### 模型参数对比

    | 特性 | Model A (Small) | Model B (Medium) | Model C (Large) |
    |------|----------------|------------------|-----------------|
    | **卷积层数** | 2 | 3 | 4 |
    | **卷积核大小** | 5×5 | 3×3 | 3×3 |
    | **滤波器数** | 32→64 | 64→128→256 | 64→128→256→512 |
    | **参数量** | ~450K | ~1.5M | ~3.2M |
    | **Dropout** | 0.25 | 0.5 | 0.5 |
    | **设计理念** | 大核快速降维 | 小核深层堆叠 | 深度+宽度 |
    """)

    st.markdown("---")

    st.markdown("""
    ### 参数调整对模型性能的影响分析

    **1. 卷积核大小的影响**

    卷积核大小直接影响模型的感受野和特征提取能力。Model A 使用 5×5 大卷积核，单层就能覆盖较大的空间范围，参数量为 5×5 = 25 个/核；而 Model B 和 C 使用 3×3 小卷积核，单层参数量仅 9 个/核。但堆叠两层 3×3 卷积可达等效 5×5 的感受野，且参数量（18 个）比单层 5×5（25 个）更少，同时引入更多非线性激活层增强特征抽象能力。因此 **多层小卷积核堆叠通常比单层大卷积核性能更优**。[reference:4]

    **2. 网络深度的影响**

    Model A（2层）→ Model B（3层）→ Model C（4层），深度递增。深层网络能提取更抽象、更语义化的特征。Model C 的 4 层结构可以捕获从边缘纹理到局部形状再到整体面部结构的多层次特征。但深度增加也带来**梯度消失风险和计算成本上升**，需要 BatchNorm 和适当的残差结构（本项目通过 AdamW + ReduceLROnPlateau 缓解）。

    **3. 滤波器数量的影响**

    Model C 在深层使用了 512 个滤波器（vs Model B 的 256），更多滤波器意味着更丰富的特征表示，但参数量也急剧增加。在 FER2013 这种中小规模数据集上，过大的模型容易过拟合——Model C 虽然理论上有更强表达能力，但可能需要更严格的 Dropout 和数据增强来防止过拟合。

    **4. Dropout 的影响**

    Model A 使用较低的 Dropout（0.25），因为其本身参数少、过拟合风险低。Model B 和 C 使用较高的 Dropout（0.5），通过随机丢弃神经元来增强泛化能力。实际上，Dropout 值的选择应与模型复杂度匹配：**模型越大、参数越多，Dropout 应适当增大**。

    **5. 训练建议**

    - 数据量较少时优先选 Model A，快速收敛
    - 追求精度选 Model B，性价比最佳
    - 有充足数据和算力时使用 Model C，但需配合强正则化
    """)

    # 如果有保存的日志，可加载对比
    st.markdown("---")
    st.subheader("📈 训练结果对比（请先训练至少两个模型）")

    log_dir = 'logs'
    if os.path.exists(log_dir):
        log_files = [f for f in os.listdir(log_dir) if f.endswith('.json')]
        if len(log_files) >= 2:
            st.info(f"找到 {len(log_files)} 个训练日志文件，可选择对比")

            selected_logs = st.multiselect("选择要对比的日志", log_files)
            if selected_logs:
                import json

                comparison_data = {}
                for log_file in selected_logs:
                    with open(os.path.join(log_dir, log_file)) as f:
                        data = json.load(f)
                        epochs_data = [d for d in data if d['type'] == 'epoch']
                        if epochs_data:
                            comparison_data[log_file] = epochs_data

                if comparison_data:
                    fig = go.Figure()
                    for name, epochs_data in comparison_data.items():
                        epochs = [d['epoch'] for d in epochs_data]
                        val_accs = [d['val_acc'] for d in epochs_data]
                        fig.add_trace(go.Scatter(x=epochs, y=val_accs, mode='lines+markers', name=name[:30]))
                    fig.update_layout(title="各模型验证准确率对比", xaxis_title='Epoch', yaxis_title='Val Accuracy',
                                      height=400, hovermode='x unified')
                    st.plotly_chart(fig, width='stretch')

    st.markdown("---")
    st.subheader("🏆 最佳模型评估图")
    model_names = ["ModelA", "ModelB", "ModelC"]
    cols = st.columns(3)
    for i, mname in enumerate(model_names):
        cm_path = f"checkpoints/{mname}_best_cm.png"
        roc_path = f"checkpoints/{mname}_best_roc.png"
        with cols[i]:
            if os.path.exists(cm_path):
                st.image(cm_path, caption=f"{mname} 混淆矩阵", width='stretch')
            else:
                st.info(f"{mname} 尚未训练")
            if os.path.exists(roc_path):
                st.image(roc_path, caption=f"{mname} ROC", width='stretch')

# ============ Tab 4: 训练日志 ============
with tab4:
    st.header("训练日志查看器")

    log_dir = 'logs'
    if os.path.exists(log_dir):
        log_files = sorted([f for f in os.listdir(log_dir) if f.endswith('.json')], reverse=True)
        if log_files:
            selected_log = st.selectbox("选择日志文件", log_files)
            if selected_log:
                with open(os.path.join(log_dir, selected_log)) as f:
                    data = json.load(f)

                st.markdown(f"### 日志文件: `{selected_log}`")
                st.markdown(f"共 {len(data)} 条记录")

                # 表格显示
                epochs_data = [d for d in data if d['type'] == 'epoch']
                if epochs_data:
                    df = pd.DataFrame(epochs_data)
                    df_display = df[
                        ['epoch', 'train_loss', 'train_acc', 'val_loss', 'val_acc', 'val_f1', 'elapsed_seconds']]
                    st.dataframe(df_display, width='stretch')

                # 文本日志
                st.markdown("### 详细日志")
                for entry in data:
                    if entry['type'] == 'start':
                        st.info(entry.get('info', ''))
                    elif entry['type'] == 'epoch':
                        st.text(
                            f"Epoch {entry['epoch']}: Train Acc={entry['train_acc']:.4f}, Val Acc={entry['val_acc']:.4f}, Val F1={entry['val_f1']:.4f}")
                    elif entry['type'] == 'message':
                        st.text(entry.get('message', ''))
        else:
            st.info("暂无训练日志，请先训练模型")
    else:
        st.info("暂无训练日志目录")

# ============ 页脚 ============
st.markdown("---")
st.markdown("*人脸情感识别系统 | FER2013 + PyTorch + Streamlit*")